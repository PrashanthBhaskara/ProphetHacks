-- Copy this to wrds_news.sql and replace the schema/table/column names with
-- the licensed WRDS news dataset your account can access.
--
-- Required named parameters:
--   %(query)s  - lexical query built from the forecast event
--   %(start)s  - beginning of the PIT lookback window
--   %(as_of)s  - forecast timestamp cutoff
--   %(limit)s  - maximum rows to return
--
-- Required output columns, or close aliases handled by the normalizer:
--   published_at, title, text, url, vendor_id, vendor_source

SELECT
  published_at,
  headline AS title,
  body AS text,
  url,
  story_id AS vendor_id,
  provider AS vendor_source
FROM your_wrds_news_schema.your_news_table
WHERE published_at >= %(start)s
  AND published_at <= %(as_of)s
  AND (
    headline ILIKE ('%%' || %(query)s || '%%')
    OR body ILIKE ('%%' || %(query)s || '%%')
  )
ORDER BY published_at DESC
LIMIT %(limit)s;
