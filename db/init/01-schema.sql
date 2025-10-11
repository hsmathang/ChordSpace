-- Minimal schema for ChordSpace experiments
-- Runs automatically when the Postgres container is initialized (docker-compose.db.yml)

CREATE TABLE IF NOT EXISTS chords (
    id SERIAL PRIMARY KEY,
    n SMALLINT NOT NULL,
    interval SMALLINT[] NOT NULL,
    notes SMALLINT[] NOT NULL,
    bass TEXT,
    octave SMALLINT,
    frequencies DOUBLE PRECISION[],
    chroma DOUBLE PRECISION[],
    tag TEXT,
    code TEXT
);

-- Unique identity to avoid duplicates when importing
CREATE UNIQUE INDEX IF NOT EXISTS idx_chords_unique_n_notes_interval
    ON chords (n, notes, interval);

