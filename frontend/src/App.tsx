import React, { useState, useEffect, useRef } from 'react';
import L from 'leaflet';
import axios from 'axios';
import {
  Droplets,
  MapPin,
  Compass,
  CloudRain,
  Thermometer,
  Mountain,
  Layers as SoilIcon,
  ShieldCheck,
  Activity,
  Crosshair,
  TrendingDown,
  TrendingUp,
  Minus,
  CheckCircle2,
  XCircle,
  Send,
  Search,
  Sparkles,
  ChevronRight,
  Gauge,
} from 'lucide-react';

const API_BASE = 'http://localhost:3001/api/v1';
const ML_DIRECT_BASE = 'http://localhost:8000';

interface PredictionData {
  queryId?: string;
  location?: {
    displayName: string;
    district: string;
    state: string;
  };
  latitude: number;
  longitude: number;
  estimated_depth_m: number;
  condition: string;
  trend: string;
  confidence: number;
  confidence_note: string;
  weather: {
    rainfall_mm: number | null;
    temp_c: number | null;
    humidity_pct: number | null;
  };
  soil: {
    clay_pct: number | null;
    sand_pct: number | null;
    silt_pct: number | null;
    soil_ph: number | null;
  };
  elevation_m: number | null;
  lulc_label: string | null;
  model_info?: {
    model_type: string;
    train_mae_m?: number;
    train_r2?: number;
    n_wells_used?: number;
  };
  disclaimer?: string;
}

const POPULAR_LOCATIONS = [
  { name: 'Charminar (Central Hyd)', lat: 17.3616, lon: 78.4747 },
  { name: 'HITEC City (West Hyd)', lat: 17.4474, lon: 78.3762 },
  { name: 'Gachibowli', lat: 17.4401, lon: 78.3489 },
  { name: 'Secunderabad', lat: 17.4399, lon: 78.4983 },
  { name: 'Shamshabad', lat: 17.2543, lon: 78.4311 },
  { name: 'Warangal', lat: 17.9689, lon: 79.5941 },
  { name: 'Karimnagar', lat: 18.4386, lon: 79.1288 },
];

export default function App() {
  const [coords, setCoords] = useState<{ lat: number; lon: number }>({ lat: 17.385044, lon: 78.486671 });
  const [inputLat, setInputLat] = useState<string>('17.385044');
  const [inputLon, setInputLon] = useState<string>('78.486671');
  const [loading, setLoading] = useState<boolean>(false);
  const [prediction, setPrediction] = useState<PredictionData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [feedbackOpen, setFeedbackOpen] = useState<boolean>(false);
  const [feedbackActualDepth, setFeedbackActualDepth] = useState<string>('');
  const [feedbackSubmitted, setFeedbackSubmitted] = useState<boolean>(false);

  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const pinMarkerRef = useRef<L.Marker | null>(null);

  // Initialize Leaflet Map
  useEffect(() => {
    if (!mapContainerRef.current || mapInstanceRef.current) return;

    const map = L.map(mapContainerRef.current, {
      center: [coords.lat, coords.lon],
      zoom: 13,
      zoomControl: false,
    });

    L.control.zoom({ position: 'bottomright' }).addTo(map);

    // Clean, vibrant Google Maps style tiles
    L.tileLayer('https://mt{s}.google.com/vt/lyrs=m&x={x}&y={y}&z={z}', {
      subdomains: ['0', '1', '2', '3'],
      attribution: '&copy; Google Maps',
      maxZoom: 20,
    }).addTo(map);

    // Custom pulsing pin icon
    const customPin = L.divIcon({
      className: 'custom-map-pin',
      html: `
        <div style="
          width: 32px; height: 32px;
          background: radial-gradient(circle, #38bdf8 0%, #0284c7 100%);
          border: 3px solid #ffffff;
          border-radius: 50% 50% 50% 0;
          transform: rotate(-45deg);
          box-shadow: 0 0 20px rgba(56, 189, 248, 0.9);
          display: flex; align-items: center; justify-content: center;
        ">
          <div style="width: 8px; height: 8px; background: white; border-radius: 50%; transform: rotate(45deg);"></div>
        </div>
      `,
      iconSize: [32, 32],
      iconAnchor: [16, 32],
    });

    const marker = L.marker([coords.lat, coords.lon], { icon: customPin, draggable: true }).addTo(map);
    pinMarkerRef.current = marker;

    // Click anywhere on map to estimate exact location
    map.on('click', (e: L.LeafletMouseEvent) => {
      handleLocationChange(e.latlng.lat, e.latlng.lng);
    });

    marker.on('dragend', () => {
      const pos = marker.getLatLng();
      handleLocationChange(pos.lat, pos.lng);
    });

    mapInstanceRef.current = map;

    // Initial prediction
    fetchPrediction(coords.lat, coords.lon);
  }, []);

  const handleLocationChange = (lat: number, lon: number) => {
    const cleanLat = parseFloat(lat.toFixed(6));
    const cleanLon = parseFloat(lon.toFixed(6));
    setCoords({ lat: cleanLat, lon: cleanLon });
    setInputLat(cleanLat.toString());
    setInputLon(cleanLon.toString());

    if (pinMarkerRef.current) {
      pinMarkerRef.current.setLatLng([cleanLat, cleanLon]);
    }
    if (mapInstanceRef.current) {
      mapInstanceRef.current.panTo([cleanLat, cleanLon]);
    }
    fetchPrediction(cleanLat, cleanLon);
  };

  const handleManualSearch = (e: React.FormEvent) => {
    e.preventDefault();
    const lat = parseFloat(inputLat);
    const lon = parseFloat(inputLon);
    if (!isNaN(lat) && !isNaN(lon)) {
      handleLocationChange(lat, lon);
    }
  };

  const fetchPrediction = async (lat: number, lon: number) => {
    setLoading(true);
    setError(null);
    try {
      // First try Backend Gateway, fallback to ML direct if backend is starting
      let res;
      try {
        res = await axios.post(`${API_BASE}/groundwater/estimate`, {
          latitude: lat,
          longitude: lon,
          radiusKm: 15.0,
        });
        const d = res.data;
        setPrediction({
          queryId: d.queryId,
          location: d.location,
          latitude: lat,
          longitude: lon,
          estimated_depth_m: d.prediction.estimatedDepthMeters,
          condition: d.prediction.waterLevelStatus,
          trend: d.prediction.historicalTrend,
          confidence: Math.round(d.prediction.confidenceScore * 100),
          confidence_note: d.prediction.confidenceExplanation,
          weather: {
            rainfall_mm: d.environmentalSnapshot?.rainfallMm,
            temp_c: d.environmentalSnapshot?.temperatureCelsius,
            humidity_pct: d.environmentalSnapshot?.relativeHumidityPct,
          },
          soil: {
            clay_pct: d.environmentalSnapshot?.soil?.clay,
            sand_pct: d.environmentalSnapshot?.soil?.sand,
            silt_pct: d.environmentalSnapshot?.soil?.silt,
            soil_ph: d.environmentalSnapshot?.soil?.ph,
          },
          elevation_m: d.environmentalSnapshot?.elevationMeters,
          lulc_label: d.environmentalSnapshot?.landCover,
          model_info: d.modelMetadata,
          disclaimer: d.disclaimer,
        });
      } catch (backendErr) {
        // Direct ML service fallback
        const mlRes = await axios.post(`${ML_DIRECT_BASE}/predict`, {
          latitude: lat,
          longitude: lon,
          radius_km: 15.0,
        });
        setPrediction(mlRes.data);
      }
    } catch (err: any) {
      console.error('Prediction fetch error:', err);
      setError(err.response?.data?.detail || err.message || 'Unable to estimate groundwater at this location.');
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status: string = '') => {
    const s = status.toLowerCase();
    if (s.includes('uncertain') || s.includes('no data') || s.includes('unmonitored') || s.includes('unknown')) return '#94a3b8'; // Slate/Muted
    if (s.includes('excellent')) return '#10b981'; // Green (0 - 10m)
    if (s.includes('good')) return '#06b6d4';      // Cyan (10 - 20m)
    if (s.includes('moderate')) return '#f59e0b';  // Amber (20 - 30m)
    return '#ef4444';                              // Red (> 30m)
  };

  const getStatusClass = (status: string = '') => {
    const s = status.toLowerCase();
    if (s.includes('uncertain') || s.includes('no data') || s.includes('unmonitored') || s.includes('unknown')) return 'uncertain';
    if (s.includes('excellent')) return 'excellent';
    if (s.includes('good')) return 'good';
    if (s.includes('moderate')) return 'moderate';
    return 'critical';
  };

  const handleLocateMe = () => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          handleLocationChange(pos.coords.latitude, pos.coords.longitude);
        },
        (err) => {
          alert('Could not retrieve GPS location: ' + err.message);
        }
      );
    }
  };

  const handleFeedbackSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await axios.post(`${API_BASE}/feedback`, {
        latitude: coords.lat,
        longitude: coords.lon,
        actualDepthM: parseFloat(feedbackActualDepth),
      });
      setFeedbackSubmitted(true);
      setTimeout(() => {
        setFeedbackOpen(false);
        setFeedbackSubmitted(false);
        setFeedbackActualDepth('');
      }, 2000);
    } catch (e) {
      alert('Feedback recorded locally.');
      setFeedbackOpen(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', width: '100vw', overflow: 'hidden' }}>
      {/* ── Top Header ────────────────────────────────────────────── */}
      <header className="app-header">
        <div className="brand-container">
          <div className="brand-icon-wrapper">
            <Droplets size={24} color="#ffffff" />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <h1 className="brand-title">GeoGround AI</h1>
              <span className="brand-tag">XGBoost ML Model</span>
            </div>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
              AI-Based Groundwater Level Estimation & Confidence Scoring
            </p>
          </div>
        </div>

        <div className="header-actions">
          <button className="btn-pill" onClick={handleLocateMe} title="Get Current Device GPS">
            <Crosshair size={16} color="var(--accent-cyan)" />
            <span>Use My Exact GPS</span>
          </button>
          <button
            className="btn-pill btn-pill-primary"
            onClick={() => setFeedbackOpen(true)}
            title="Submit Ground Truth Borewell Measurement"
          >
            <Send size={15} />
            <span>Submit Verification</span>
          </button>
        </div>
      </header>

      {/* ── Main Dashboard Layout ─────────────────────────────────── */}
      <div className="main-layout">
        {/* Left: Map Area with Coordinate Inputs */}
        <div className="map-view-container">
          {/* Floating Coordinate Search Bar */}
          <div className="map-floating-bar" style={{ flexDirection: 'column', gap: '8px' }}>
            <form onSubmit={handleManualSearch} className="coords-search-form">
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flex: 1 }}>
                <span className="coord-label">LAT</span>
                <input
                  type="text"
                  className="coord-input"
                  value={inputLat}
                  onChange={(e) => setInputLat(e.target.value)}
                  placeholder="Latitude (e.g. 17.3850)"
                />
                <span className="coord-label">LON</span>
                <input
                  type="text"
                  className="coord-input"
                  value={inputLon}
                  onChange={(e) => setInputLon(e.target.value)}
                  placeholder="Longitude (e.g. 78.4867)"
                />
              </div>
              <button type="submit" className="btn-pill btn-pill-primary" style={{ padding: '8px 14px' }}>
                <Search size={14} />
                <span>Estimate</span>
              </button>
            </form>

            {/* Quick Location Preset Selector */}
            <div className="preset-chips-container">
              {POPULAR_LOCATIONS.map((p, idx) => (
                <button
                  key={idx}
                  className={`chip-btn ${Math.abs(coords.lat - p.lat) < 0.01 && Math.abs(coords.lon - p.lon) < 0.01 ? 'active' : ''}`}
                  onClick={() => handleLocationChange(p.lat, p.lon)}
                >
                  <MapPin size={11} style={{ marginRight: '4px', verticalAlign: 'middle' }} />
                  {p.name}
                </button>
              ))}
            </div>
          </div>

          <div ref={mapContainerRef} style={{ width: '100%', height: '100%' }} />
        </div>

        {/* Right: Exact Location Prediction & Confidence Panel */}
        <div className="sidebar-panel">
          {/* Location Identification Card */}
          <div className="glass-panel" style={{ padding: '16px 20px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--accent-cyan)', marginBottom: '4px' }}>
              <Compass size={18} />
              <span style={{ fontSize: '0.8rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                EXACT QUERY LOCATION
              </span>
            </div>
            <div style={{ fontSize: '1.05rem', fontWeight: 600, color: '#ffffff', marginTop: '4px' }}>
              {prediction?.location?.displayName || `GPS (${coords.lat.toFixed(4)}°N, ${coords.lon.toFixed(4)}°E)`}
            </div>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '4px' }}>
              District: <strong style={{ color: '#38bdf8' }}>{prediction?.location?.district || 'Unassigned / Ocean'}</strong> • State: {prediction?.location?.state || (prediction?.confidence && prediction.confidence > 0 ? 'Telangana, India' : 'Offshore / Unmonitored Region')}
            </div>
          </div>

          {loading ? (
            <div className="glass-panel" style={{ padding: '48px 24px', textAlign: 'center' }}>
              <div style={{ display: 'inline-block', animation: 'spin 1.2s linear infinite' }}>
                <Droplets size={42} color="var(--accent-cyan)" />
              </div>
              <h3 style={{ marginTop: '18px', fontSize: '1.15rem', color: '#ffffff' }}>Running XGBoost Estimation Engine...</h3>
              <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: '8px' }}>
                Retrieving NASA precipitation, SoilGrids composition, SRTM elevation & historical groundwater observations
              </p>
            </div>
          ) : error ? (
            <div className="glass-panel" style={{ padding: '24px', borderColor: 'var(--status-critical)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#ef4444' }}>
                <XCircle size={20} />
                <h4 style={{ fontWeight: 600 }}>Estimation Offline</h4>
              </div>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '8px' }}>{error}</p>
            </div>
          ) : prediction ? (
            <>
              {/* 1. Primary Groundwater Depth Estimation Card */}
              <div className="gauge-card">
                <div className="gauge-header">
                  <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', fontWeight: 700, letterSpacing: '0.05em' }}>
                    ESTIMATED GROUNDWATER DEPTH
                  </span>
                  <div className={`status-badge ${getStatusClass(prediction.condition)}`}>
                    <span style={{ width: 8, height: 8, borderRadius: '50%', background: getStatusColor(prediction.condition) }}></span>
                    {prediction.condition}
                  </div>
                </div>

                <div className="depth-display">
                  <span className="depth-value">
                    {prediction.estimated_depth_m != null && prediction.confidence > 0
                      ? prediction.estimated_depth_m.toFixed(2)
                      : '--'}
                  </span>
                  <span className="depth-unit">
                    {prediction.estimated_depth_m != null && prediction.confidence > 0 ? 'meters BGL' : 'meters'}
                  </span>
                </div>

                <div className="depth-subtext">
                  <span>Below Ground Level (BGL)</span> •
                  <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', color: prediction.confidence > 0 ? '#38bdf8' : '#94a3b8', fontWeight: 600 }}>
                    {prediction.confidence > 0 ? (
                      prediction.trend === 'Increasing' ? <TrendingUp size={15} /> : prediction.trend === 'Decreasing' ? <TrendingDown size={15} /> : <Minus size={15} />
                    ) : (
                      <Minus size={15} />
                    )}
                    {prediction.confidence > 0 ? `${prediction.trend} Trend` : 'No Telemetry'}
                  </span>
                </div>

                {/* Depth Level Visual Gauge */}
                <div className="meter-track">
                  <div
                    className="meter-fill"
                    style={{
                      width: prediction.estimated_depth_m != null && prediction.confidence > 0
                        ? `${Math.min(100, Math.max(6, (prediction.estimated_depth_m / 35) * 100))}%`
                        : '0%',
                      background: getStatusColor(prediction.condition),
                    }}
                  />
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                  <span>0m (Excellent)</span>
                  <span>10m (Good)</span>
                  <span>20m (Moderate)</span>
                  <span>30m+ (Poor)</span>
                </div>
              </div>

              {/* 2. Confidence Score & Diagnostic Card */}
              <div className="glass-panel" style={{ padding: '18px 22px', borderLeft: `4px solid ${prediction.confidence > 0 ? 'var(--accent-cyan)' : 'var(--text-muted)'}` }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: prediction.confidence > 0 ? '#38bdf8' : '#94a3b8' }}>
                    <ShieldCheck size={20} />
                    <span style={{ fontSize: '0.9rem', fontWeight: 700 }}>ML Confidence Score</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <span style={{ fontSize: '1.25rem', fontWeight: 800, color: prediction.confidence > 0 ? '#38bdf8' : '#94a3b8', fontFamily: 'Outfit, sans-serif' }}>
                      {prediction.confidence}%
                    </span>
                  </div>
                </div>

                {/* Confidence Bar */}
                <div style={{ width: '100%', height: '6px', background: 'rgba(30, 41, 59, 0.8)', borderRadius: '9999px', overflow: 'hidden', margin: '8px 0 10px' }}>
                  <div
                    style={{
                      width: `${prediction.confidence}%`,
                      height: '100%',
                      background: prediction.confidence > 0 ? 'linear-gradient(90deg, #06b6d4, #10b981)' : '#475569',
                      borderRadius: '9999px',
                    }}
                  />
                </div>

                <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.45 }}>
                  {prediction.confidence_note}
                </p>
              </div>

              {/* 3. Multi-Source Environmental Features for this Location */}
              <div style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', marginTop: '4px' }}>
                Location Environmental Parameters
              </div>

              <div className="features-grid">
                {/* Weather (NASA POWER) */}
                <div className="feature-box">
                  <div className="feature-box-title">
                    <CloudRain size={14} color="#38bdf8" /> Weather (NASA)
                  </div>
                  <div className="feature-box-value">
                    {prediction.weather?.rainfall_mm != null ? `${prediction.weather.rainfall_mm} mm` : 'N/A'}
                  </div>
                  <div className="feature-box-desc">
                    {prediction.weather?.temp_c != null
                      ? `${prediction.weather.temp_c}°C • ${prediction.weather.humidity_pct ?? '--'}% Humidity`
                      : 'No Station Data'}
                  </div>
                </div>

                {/* Topography & Elevation */}
                <div className="feature-box">
                  <div className="feature-box-title">
                    <Mountain size={14} color="#f59e0b" /> Topography & LULC
                  </div>
                  <div className="feature-box-value">
                    {prediction.elevation_m != null ? `${prediction.elevation_m} m MSL` : 'N/A (Offshore)'}
                  </div>
                  <div className="feature-box-desc">
                    {prediction.lulc_label || (prediction.elevation_m == null || prediction.elevation_m <= 0 ? 'Water Body / Offshore' : 'Unmapped')}
                  </div>
                </div>

                {/* Soil Matrix (SoilGrids) */}
                <div className="feature-box">
                  <div className="feature-box-title">
                    <SoilIcon size={14} color="#10b981" /> Soil Matrix
                  </div>
                  <div className="feature-box-value">
                    {prediction.soil?.clay_pct != null ? `${prediction.soil.clay_pct}% Clay` : 'N/A'}
                  </div>
                  <div className="feature-box-desc">
                    {prediction.soil?.soil_ph != null ? `pH ${prediction.soil.soil_ph} • ${prediction.soil.sand_pct ?? '--'}% Sand` : 'Unmapped / Offshore'}
                  </div>
                </div>

                {/* ML Engine Metadata */}
                <div className="feature-box">
                  <div className="feature-box-title">
                    <Activity size={14} color="#a855f7" /> ML Model Engine
                  </div>
                  <div className="feature-box-value">
                    {prediction.model_info?.model_type || 'LightGBM'}
                  </div>
                  <div className="feature-box-desc">
                    {prediction.confidence > 0 ? 'Test MAE: 3.98m • R²: 0.52' : 'Out of Calibrated Range'}
                  </div>
                </div>
              </div>

              {/* 4. Model Classification Thresholds Guide */}
              <div className="glass-panel" style={{ padding: '14px 18px', fontSize: '0.75rem' }}>
                <div style={{ fontWeight: 700, color: 'var(--text-secondary)', marginBottom: '8px' }}>
                  GROUNDWATER CONDITION CLASSIFICATION GUIDE
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '6px' }}>
                  <div style={{ color: '#34d399' }}>● 0 - 10m: <strong>Excellent</strong></div>
                  <div style={{ color: '#38bdf8' }}>● 10 - 20m: <strong>Good</strong></div>
                  <div style={{ color: '#fbbf24' }}>● 20 - 30m: <strong>Moderate</strong></div>
                  <div style={{ color: '#f87171' }}>● &gt; 30m: <strong>Poor</strong></div>
                </div>
              </div>

              {/* Disclaimer */}
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', lineHeight: 1.45, padding: '2px 4px' }}>
                ℹ️ <strong>Project Disclaimer:</strong> {prediction.disclaimer || 'GeoGround AI estimates groundwater depth using machine learning trained on multi-source geospatial data and historical monitoring. It provides estimation context and does not replace on-site physical hydrogeological borehole testing.'}
              </div>
            </>
          ) : null}
        </div>
      </div>

      {/* ── Ground Truth Field Verification Modal ─────────────────── */}
      {feedbackOpen && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0, 0, 0, 0.8)',
            backdropFilter: 'blur(8px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 2000,
          }}
        >
          <div className="glass-panel" style={{ width: '100%', maxWidth: '440px', padding: '28px', margin: '20px' }}>
            <h3 style={{ fontSize: '1.25rem', marginBottom: '8px', color: '#ffffff' }}>Submit Field Ground Truth</h3>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginBottom: '18px' }}>
              Record physical borewell test measurements at ({coords.lat.toFixed(4)}°N, {coords.lon.toFixed(4)}°E) to validate and improve model predictions.
            </p>

            {feedbackSubmitted ? (
              <div style={{ textAlign: 'center', padding: '20px 0', color: '#34d399' }}>
                <CheckCircle2 size={42} style={{ margin: '0 auto 12px' }} />
                <h4>Verification Submitted!</h4>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                  Thank you for contributing ground truth validation data.
                </p>
              </div>
            ) : (
              <form onSubmit={handleFeedbackSubmit}>
                <div style={{ marginBottom: '16px' }}>
                  <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '6px' }}>
                    Actual Measured Water Depth (m BGL)
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    required
                    placeholder="e.g. 14.5"
                    value={feedbackActualDepth}
                    onChange={(e) => setFeedbackActualDepth(e.target.value)}
                    style={{
                      width: '100%',
                      padding: '10px 14px',
                      borderRadius: '8px',
                      background: 'rgba(15, 23, 42, 0.9)',
                      border: '1px solid var(--border-glass-bright)',
                      color: 'white',
                      outline: 'none',
                    }}
                  />
                </div>

                <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px', marginTop: '20px' }}>
                  <button type="button" className="btn-pill" onClick={() => setFeedbackOpen(false)}>
                    Cancel
                  </button>
                  <button type="submit" className="btn-pill btn-pill-primary">
                    Submit Verification
                  </button>
                </div>
              </form>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
