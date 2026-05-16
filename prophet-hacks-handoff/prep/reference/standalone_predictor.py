#!/usr/bin/env python3
"""
Standalone prediction script that can run predictions on event data from CSV files.

This script is self-contained and doesn't depend on the app module.
It reads event data from CSV (from retrieve_test_events.py) and runs LLM predictions.

Usage:
    # Run all events
    python3 standalone_predictor.py --input_csv test_dataset_100.csv --output_csv predictions.csv --base_url URL_ENDPOINT --api_key API_KEY --model MODEL_NAME --run_all
    
    # Run specific events
    python3 standalone_predictor.py --input_csv test_dataset_100.csv --output_csv predictions.csv --base_url URL_ENDPOINT --api_key API_KEY --model MODEL_NAME --run_specific KXATPMATCH-25JUL02SHEHIJ
"""

import json
import asyncio
import argparse
import pandas as pd
import re
import ast
from typing import List, Dict
from dataclasses import dataclass
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential


def normalize_market_text(text: str) -> str:
    """
    Normalize market text by converting ASCII apostrophes to Unicode right single quotation marks.
    This ensures consistency between LLM responses and database storage.
    """
    return text.replace("'", "\u2019")


@dataclass
class MarketPrediction:
    market: str
    probability: float


@dataclass
class PredictionOutput:
    probabilities: List[MarketPrediction]
    rationale: str


class PredictionPrompts:
    """Prompts for market prediction tasks."""

    @staticmethod
    def create_task_prompt(event_title: str, market_names: List[str]) -> str:
        """Create the task prompt for market prediction."""
        market_list_str = "\n".join([f"- {market}" for market in market_names])
        json_example = ",\n                ".join(
            [f'"{market}": <probability_value_from_0_to_1>' for market in market_names])

        return f"""
                You are an AI assistant specialized in analyzing and predicting real-world events. 
                You have deep expertise in predicting the outcome of the event: "{event_title}"

                Note that this event occurs in the future. You will be given a list of sources with their summaries, rankings, and expert comments.
                Based on these collected sources, your goal is to extract meaningful insights and provide well-reasoned predictions based on the given data.
                You will be predicting the probability (as a float value from 0 to 1) of ONLY the following possible outcomes:
                {market_list_str}

                IMPORTANT CONSTRAINTS:
                1. You MUST ONLY provide probabilities for the exact possible outcomes listed above
                2. Do NOT create or invent any additional outcomes
                3. Use exactly the same outcome names as provided (case-sensitive)
                4. Ensure all probabilities are between 0 and 1

                Your response MUST be in JSON format with the following structure:
                ```json
                {{
                    "rationale": "<text_explaining_your_rationale>",
                    "probabilities": {{
                        {json_example}
                    }}
                }}
                ```

                In the rationale section of your response, please provide a short, concise, 3 sentence rationale that explains:
                - How you weighed different pieces of information
                - Your reasoning for the probability distribution you assigned
                - Any key factors or uncertainties you considered
                """.strip()

    @staticmethod
    def create_user_prompt(sources: str, market_stats: dict = None) -> str:
        """Create the user prompt for providing source data."""
        market_stats_info = ""
        if market_stats:
            market_stats_info = f"""
            CURRENT ONLINE TRADING DATA:
            You also have access to the predicted outcome probability (last trading price of each outcome turned out to be yes) from a popular prediction market at the moment of your prediction:
            {json.dumps(market_stats, indent=2)}
            
            Note: Market data can provide insights into the current consensus of the market influenced by traders of various beliefs and private information. However, you should not rely on market data alone to make your prediction.
            Please consider both the market data and the information sources to help you make a well-calibrated prediction. 
            """
        return f"""
                HERE IS THE GIVEN DATA: it is a list of sources with their summaries, rankings, and user comments. 
                The smaller the ranking number, the more you should weight the source in your prediction. 
                {sources} 
                {market_stats_info}
                """.strip()


class LLMError(Exception):
    """Exception raised for LLM-related errors."""
    pass


class StandalonePredictor:
    """Self-contained predictor that doesn't depend on app modules."""

    def __init__(self, model: str = "minimax/minimax-m1", 
                 base_url: str = "https://openrouter.ai/api/v1", 
                 api_key: str = None, 
                 reasoning: str = None):
        if not api_key:
            raise ValueError("API key is required")
            
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=800
        )
        self.base_url = base_url
        self.model = model
        self.reasoning = reasoning

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
    def completion_with_backoff(self, **kwargs):
        return self.client.chat.completions.create(**kwargs)

    async def async_completion_with_backoff(self, **kwargs):
        """Async wrapper for completion with backoff."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.completion_with_backoff(**kwargs)
        )

    def _validate_response(self, dynamic_out: dict, expected_markets: list) -> tuple[bool, str]:
        """Validate the model response for hallucination and market mismatch."""
        if not isinstance(dynamic_out, dict):
            return False, "Response is not a valid dictionary"

        if "probabilities" not in dynamic_out:
            return False, "Missing 'probabilities' field in response"

        if "rationale" not in dynamic_out:
            return False, "Missing 'rationale' field in response"

        probs = dynamic_out["probabilities"]
        if not isinstance(probs, dict):
            return False, "Probabilities field is not a dictionary"

        # Check for market mismatch
        response_markets = set(normalize_market_text(market) for market in probs.keys())
        expected_markets_set = set(normalize_market_text(market) for market in expected_markets)

        extra_markets = response_markets - expected_markets_set
        if extra_markets:
            return False, f"Model hallucinated extra markets: {list(extra_markets)}"

        missing_markets = expected_markets_set - response_markets
        if missing_markets:
            return False, f"Model failed to provide probabilities for markets: {list(missing_markets)}"

        # Validate probability values
        for market, prob in probs.items():
            if not isinstance(prob, (int, float)):
                return False, f"Invalid probability type for {market}: {type(prob)}"
            if not (0 <= prob <= 1):
                return False, f"Probability for {market} out of range [0,1]: {prob}"

        return True, ""

    async def predict_event_async(self, event_title: str, markets: List[str], sources: List[Dict], 
                                 market_stats: Dict = None) -> PredictionOutput:
        """Run prediction for a single event (async version)."""
        sources_text = self._format_sources(sources)
        task_prompt = PredictionPrompts.create_task_prompt(event_title, markets)
        user_input = PredictionPrompts.create_user_prompt(sources_text, market_stats)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                messages = [
                    {"role": "system", "content": task_prompt},
                    {"role": "user", "content": user_input}
                ]

                if attempt > 0:
                    messages.append({
                        "role": "user",
                        "content": "The previous response had validation errors. Please provide a valid response with exactly the specified markets and proper JSON format."
                    })

                payload = dict(model=self.model, messages=messages)
                if self.reasoning and self.base_url == "https://openrouter.ai/api/v1":
                    payload["extra_body"] = dict(reasoning={"effort": self.reasoning})
                elif self.reasoning:
                    payload["reasoning"] = {"effort": self.reasoning}

                chat_completion = await self.async_completion_with_backoff(**payload)
                raw_content = chat_completion.choices[0].message.content
                
                # Extract JSON from response
                json_start = raw_content.find('{')
                if json_start > 0:
                    raw_content = raw_content[json_start:]
                
                json_end = raw_content.rfind('}')
                if json_end != -1 and json_end < len(raw_content) - 1:
                    raw_content = raw_content[:json_end + 1]
                
                dynamic_out = json.loads(raw_content)
                
                if dynamic_out is None:
                    if attempt < max_retries - 1:
                        continue
                    raise LLMError("Model refused to provide a prediction or returned invalid format")

                # Validate response
                is_valid, error_msg = self._validate_response(dynamic_out, markets)
                if not is_valid:
                    print(f"Validation error on attempt {attempt + 1}: {error_msg}")
                    if attempt < max_retries - 1:
                        continue
                    else:
                        raise LLMError(f"Validation failed after {max_retries} attempts: {error_msg}")

                flat_probs: dict[str, float] = dynamic_out["probabilities"]
                preds = [
                    MarketPrediction(market=mk, probability=prob)
                    for mk, prob in flat_probs.items()
                ]

                return PredictionOutput(
                    probabilities=preds,
                    rationale=dynamic_out["rationale"]
                )

            except json.JSONDecodeError as e:
                print(f"JSON decode error on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    continue
                raise LLMError(f"Failed to parse JSON response after {max_retries} attempts: {str(e)}")
            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    continue
                raise LLMError(f"Failed to get prediction: {str(e)}")

    def predict_event(self, event_title: str, markets: List[str], sources: List[Dict], 
                     market_stats: Dict = None) -> PredictionOutput:
        """Run prediction for a single event (sync version)."""
        return asyncio.run(self.predict_event_async(event_title, markets, sources, market_stats))

    def _format_sources(self, sources: List[Dict]) -> str:
        """Format sources list into a string for the prompt."""
        if not sources:
            return "No sources available for this event."
        
        formatted_sources = []
        for source in sources:
            source_text = f"Source {source.get('ranking', 'N/A')}: {source.get('title', 'No title')}\n"
            source_text += f"URL: {source.get('url', 'No URL')}\n"
            source_text += f"Summary: {source.get('summary', 'No summary')}\n"
            formatted_sources.append(source_text)
        
        return "\n---\n".join(formatted_sources)


async def process_events_async(events_data: List[Dict], predictor: StandalonePredictor) -> List[Dict]:
    """Process multiple events asynchronously."""
    
    async def process_single_event(event_data: Dict) -> Dict:
        try:
            print(f"Processing event: {event_data['event_ticker']}")
            
            # Parse markets with debug info
            try:
                markets = event_data['markets']
                print(f"Raw markets data: {repr(markets[:100])}")  # Debug: show first 100 chars
                if isinstance(markets, str):
                    markets = json.loads(markets)
                print(f"Parsed markets: {markets}")
            except json.JSONDecodeError as e:
                print(f"Error parsing markets: {e}")
                raise
            
            try:
                sources = event_data.get('sources', [])
                if isinstance(sources, str):
                    sources_str = re.sub(r"UUID\('([^']+)'\)", r"'\1'", sources)
                    sources = ast.literal_eval(sources_str)
                    
                print(f"Parsed sources count: {len(sources)}")
            except (json.JSONDecodeError, ValueError, SyntaxError) as e:
                print(f"Error parsing sources: {e}")
                raise
            
            # Parse market_info and extract only last_price, yes_ask, no_ask (like the real app)
            market_stats = None
            if 'market_info' in event_data and event_data['market_info']:
                try:
                    market_info_raw = event_data['market_info']
                    print(f"Raw market_info data: {repr(market_info_raw[:200])}")  # Debug
                    if isinstance(market_info_raw, str):
                        market_info = ast.literal_eval(market_info_raw)
                    else:
                        market_info = market_info_raw
                    
                    # Extract only the fields that the real app uses: last_price, yes_ask, no_ask
                    if market_info:
                        market_stats = {}
                        for market_title, market_data in market_info.items():
                            market_stats[market_title] = {
                                "last_price": market_data.get("last_price"),
                                "yes_ask": market_data.get("yes_ask"), 
                                "no_ask": market_data.get("no_ask")
                            }
                    print(f"Extracted market_stats keys: {list(market_stats.keys()) if market_stats else None}")
                except (ValueError, SyntaxError, TypeError) as e:
                    print(f"Error parsing market_info (non-fatal): {e}")
                    pass
            
            # Run prediction
            prediction = await predictor.predict_event_async(
                event_title=event_data['title'],
                markets=markets,
                sources=sources,
                market_stats=market_stats
            )
            
            complete_prediction = {
                'probabilities': [{'market': pred.market, 'probability': pred.probability} 
                                for pred in prediction.probabilities],
                'rationale': prediction.rationale
            }
            
            result = {
                'event_ticker': event_data['event_ticker'],
                'title': event_data['title'],
                'category': event_data.get('category', ''),
                'markets': json.dumps(markets),
                'prediction': json.dumps(complete_prediction),  # Store entire response as JSON
                'model': predictor.model,
                'status': 'success'
            }
                
            print(f"Successfully processed {event_data['event_ticker']}")
            return result
            
        except Exception as e:
            print(f"✗ Error processing {event_data['event_ticker']}: {str(e)}")
            return {
                'event_ticker': event_data['event_ticker'],
                'title': event_data['title'],
                'category': event_data.get('category', ''),
                'markets': json.dumps(markets) if 'markets' in locals() else '',
                'prediction': '',
                'rationale': '',
                'model': predictor.model,
                'status': 'error',
                'error_message': str(e)
            }
    
    # Process events concurrently
    tasks = [process_single_event(event) for event in events_data]
    results = await asyncio.gather(*tasks)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Standalone event prediction script")
    parser.add_argument("--input_csv", "-i", required=True, 
                       help="Input CSV file with event data (from retrieve_test_events.py)")
    parser.add_argument("--output_csv", "-o", required=True, 
                       help="Output CSV file for predictions")
    parser.add_argument("--api_key", "-k", required=True, 
                       help="OpenRouter API key")
    parser.add_argument("--base_url", "-u", default="https://openrouter.ai/api/v1", 
                       help="API base URL (default: https://openrouter.ai/api/v1)")
    parser.add_argument("--model", "-m", default="minimax/minimax-m1", 
                       help="Model to use for predictions")
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--run_all", action="store_true", 
                           help="Run predictions for all events in the CSV")
    mode_group.add_argument("--run_specific", type=str, 
                           help="Run predictions for specific event tickers (comma-separated)")
    
    args = parser.parse_args()
    
    # Read the input CSV
    print(f"Reading events from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    print(f"Found {len(df)} events")
    
    # Filter events based on mode
    if args.run_specific:
        event_tickers = [ticker.strip() for ticker in args.run_specific.split(',')]
        df = df[df['event_ticker'].isin(event_tickers)]
        print(f"Filtered to {len(df)} specific events: {event_tickers}")
        
        if df.empty:
            print("No matching events found!")
            return
    
    # Initialize predictor
    predictor = StandalonePredictor(
        model=args.model, 
        base_url=args.base_url, 
        api_key=args.api_key
    )
    
    # Convert DataFrame to list of dicts
    events_data = df.to_dict('records')
    
    # Process events
    if args.run_all and len(events_data) > 1:
        print("Running async processing for multiple events...")
        results = asyncio.run(process_events_async(events_data, predictor))
    else:
        print("Processing...")
        results = []
        for i, event_data in enumerate(events_data):
            print(f"Processing event {i+1}/{len(events_data)}: {event_data['event_ticker']}")
            result = asyncio.run(process_events_async([event_data], predictor))
            results.extend(result)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_csv, index=False)
    
    # Summary
    successful = len([r for r in results if r['status'] == 'success'])
    print(f"\nResults saved to {args.output_csv}")
    print(f"Successfully processed: {successful}/{len(results)} events")


if __name__ == "__main__":
    main()