-- GeoGround AI Database Schema Initialization (PostgreSQL)
-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. Districts Lookup & Health Stats
CREATE TABLE IF NOT EXISTS districts (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    state VARCHAR(100) DEFAULT 'Telangana',
    total_monitoring_wells INTEGER DEFAULT 0,
    average_depth_m NUMERIC(5, 2),
    water_condition_status VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. Monitoring Wells Static Spatial Table
CREATE TABLE IF NOT EXISTS monitoring_wells (
    id SERIAL PRIMARY KEY,
    well_name VARCHAR(150) UNIQUE NOT NULL,
    agency VARCHAR(100),
    state VARCHAR(100) DEFAULT 'Telangana',
    district VARCHAR(100),
    tehsil VARCHAR(100),
    block VARCHAR(100),
    village VARCHAR(100),
    latitude NUMERIC(9, 6) NOT NULL,
    longitude NUMERIC(9, 6) NOT NULL,
    elevation_msl NUMERIC(7, 2),
    n_observations INTEGER DEFAULT 0,
    mean_depth_m NUMERIC(5, 2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_wells_lat_lon ON monitoring_wells(latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_wells_district ON monitoring_wells(district);

-- 3. Prediction Log & Location Cache
CREATE TABLE IF NOT EXISTS prediction_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    latitude NUMERIC(9, 6) NOT NULL,
    longitude NUMERIC(9, 6) NOT NULL,
    search_radius_km NUMERIC(5, 2) DEFAULT 15.0,
    estimated_depth_m NUMERIC(5, 2) NOT NULL,
    condition_label VARCHAR(50) NOT NULL,
    trend_label VARCHAR(50) NOT NULL,
    confidence_score NUMERIC(4, 3) NOT NULL,
    confidence_note TEXT,
    rainfall_mm NUMERIC(7, 2),
    temperature_c NUMERIC(5, 2),
    relative_humidity_pct NUMERIC(5, 2),
    elevation_m NUMERIC(7, 2),
    land_use_cover VARCHAR(100),
    soil_clay_pct NUMERIC(5, 2),
    soil_sand_pct NUMERIC(5, 2),
    soil_silt_pct NUMERIC(5, 2),
    soil_ph NUMERIC(4, 2),
    nearest_well_name VARCHAR(150),
    nearest_well_distance_km NUMERIC(6, 2),
    nearby_wells_count INTEGER DEFAULT 0,
    user_ip VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_predictions_lat_lon ON prediction_logs(latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_predictions_created ON prediction_logs(created_at DESC);

-- 4. User Feedback / Field Verification Reports
CREATE TABLE IF NOT EXISTS field_feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    prediction_id UUID REFERENCES prediction_logs(id) ON DELETE SET NULL,
    latitude NUMERIC(9, 6) NOT NULL,
    longitude NUMERIC(9, 6) NOT NULL,
    actual_measured_depth_m NUMERIC(5, 2),
    user_comment TEXT,
    water_yield_rating VARCHAR(50), -- High, Medium, Low, Dry
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
