-- PIT-safe WRDS news query for licensed RavenPack/Dow Jones news analytics.
--
-- vendor_evidence.py renders {year} from the forecast as_of timestamp before
-- sending this SQL to WRDS. The connector supplies:
--   %(query)s  event/entity search text
--   %(start)s  beginning of lookback window
--   %(as_of)s  forecast timestamp cutoff
--   %(limit)s  max rows to return

WITH candidates AS (
  SELECT
    'ravenpack_dj.rpa_djpr_global_macro_{year}' AS vendor_source_table,
    timestamp_utc,
    rpa_date_utc,
    rp_story_id,
    headline,
    event_text,
    entity_name,
    source_name,
    topic,
    "group" AS event_group,
    type AS event_type,
    category,
    relevance,
    event_relevance,
    event_sentiment_score
  FROM ravenpack_dj.rpa_djpr_global_macro_{year}
  WHERE rpa_date_utc >= CAST(%(start)s AS date)
    AND rpa_date_utc <= CAST(%(as_of)s AS date)
    AND timestamp_utc >= CAST(%(start)s AS timestamp)
    AND timestamp_utc <= CAST(%(as_of)s AS timestamp)

  UNION ALL

  SELECT
    'ravenpack_dj.rpa_djpr_equities_{year}' AS vendor_source_table,
    timestamp_utc,
    rpa_date_utc,
    rp_story_id,
    headline,
    event_text,
    entity_name,
    source_name,
    topic,
    "group" AS event_group,
    type AS event_type,
    category,
    relevance,
    event_relevance,
    event_sentiment_score
  FROM ravenpack_dj.rpa_djpr_equities_{year}
  WHERE rpa_date_utc >= CAST(%(start)s AS date)
    AND rpa_date_utc <= CAST(%(as_of)s AS date)
    AND timestamp_utc >= CAST(%(start)s AS timestamp)
    AND timestamp_utc <= CAST(%(as_of)s AS timestamp)
),
matched AS (
  SELECT *
  FROM candidates
  WHERE headline ILIKE ('%%' || %(query)s || '%%')
    OR entity_name ILIKE ('%%' || %(query)s || '%%')
    OR event_text ILIKE ('%%' || %(query)s || '%%')
    OR topic ILIKE ('%%' || %(query)s || '%%')
    OR category ILIKE ('%%' || %(query)s || '%%')
)
SELECT
  timestamp_utc AS published_at,
  headline AS title,
  concat_ws(
    ' | ',
    event_text,
    'entity=' || entity_name,
    'topic=' || topic,
    'group=' || event_group,
    'type=' || event_type,
    'category=' || category,
    'sentiment=' || event_sentiment_score::text,
    'relevance=' || relevance::text,
    'event_relevance=' || event_relevance::text
  ) AS text,
  NULL::text AS url,
  rp_story_id AS vendor_id,
  source_name AS vendor_source,
  topic,
  category,
  relevance,
  vendor_source_table
FROM matched
ORDER BY
  COALESCE(event_relevance, 0) DESC,
  COALESCE(relevance, 0) DESC,
  timestamp_utc DESC
LIMIT %(limit)s;
