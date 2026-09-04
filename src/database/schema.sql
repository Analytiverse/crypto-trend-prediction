CREATE TABLE IF NOT EXISTS coins (
    coin_id VARCHAR(100) PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    name VARCHAR(100) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


CREATE TABLE IF NOT EXISTS market_history_raw (
    id BIGSERIAL PRIMARY KEY,
    coin_id VARCHAR(100) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    price NUMERIC,
    market_cap NUMERIC,
    total_volume NUMERIC,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_market_history_raw_coin
        FOREIGN KEY (coin_id)
        REFERENCES coins(coin_id)
        ON DELETE CASCADE,

    CONSTRAINT uq_market_history_raw_coin_timestamp
        UNIQUE (coin_id, timestamp)
);


CREATE TABLE IF NOT EXISTS market_hourly (
    coin_id VARCHAR(100) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    price NUMERIC NOT NULL,
    market_cap NUMERIC,
    total_volume NUMERIC,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT pk_market_hourly
        PRIMARY KEY (coin_id, timestamp),

    CONSTRAINT fk_market_hourly_coin
        FOREIGN KEY (coin_id)
        REFERENCES coins(coin_id)
        ON DELETE CASCADE
);


CREATE TABLE IF NOT EXISTS market_snapshots (
    id BIGSERIAL PRIMARY KEY,
    coin_id VARCHAR(100) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,

    current_price NUMERIC,
    market_cap NUMERIC,
    market_cap_rank INTEGER,
    total_volume NUMERIC,

    high_24h NUMERIC,
    low_24h NUMERIC,

    price_change_24h NUMERIC,
    price_change_percentage_24h NUMERIC,

    circulating_supply NUMERIC,
    total_supply NUMERIC,
    max_supply NUMERIC,

    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_market_snapshots_coin
        FOREIGN KEY (coin_id)
        REFERENCES coins(coin_id)
        ON DELETE CASCADE,

    CONSTRAINT uq_market_snapshots_coin_timestamp
        UNIQUE (coin_id, timestamp)
);


CREATE INDEX IF NOT EXISTS idx_market_history_raw_timestamp
    ON market_history_raw(timestamp);

CREATE INDEX IF NOT EXISTS idx_market_hourly_timestamp
    ON market_hourly(timestamp);

CREATE INDEX IF NOT EXISTS idx_market_snapshots_timestamp
    ON market_snapshots(timestamp);