-- Quick schema validation for the chords table expected by ChordSpace.
-- Run with: psql -f tools/sql/check_schema.sql

\pset format aligned
\pset linestyle unicode

SELECT
    column_name,
    data_type,
    udt_name,
    is_nullable,
    column_default
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name = 'chords'
ORDER BY ordinal_position;

SELECT
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'public'
  AND tablename = 'chords'
ORDER BY indexname;
