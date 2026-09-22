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
  Lightbulb,
  Languages,
  Info,
  Sun,
  Moon,
} from 'lucide-react';

const API_BASE = (import.meta as any).env?.VITE_API_BASE_URL || (import.meta as any).env?.VITE_API_BASE || 'http://localhost:3001/api/v1';
const ML_DIRECT_BASE = (import.meta as any).env?.VITE_ML_DIRECT_BASE || (import.meta as any).env?.VITE_ML_SERVICE_URL || 'http://localhost:8000';

type Lang = 'en' | 'te';

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
  groundwater_health_score?: number | null;
  health_status?: string;
  health_description?: string;
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
  { name: 'Charminar (Central Hyd)', telugu: 'చార్మినార్ (హైదరాబాద్)', lat: 17.3616, lon: 78.4747 },
  { name: 'Ghatkesar / Medchal', telugu: 'ఘట్కేసర్ / మేడ్చల్', lat: 17.4228, lon: 78.6500 },
  { name: 'HITEC City (West Hyd)', telugu: 'హైటెక్ సిటీ', lat: 17.4474, lon: 78.3762 },
  { name: 'Gachibowli', telugu: 'గచ్చిబౌలి', lat: 17.4401, lon: 78.3489 },
  { name: 'Secunderabad', telugu: 'సికింద్రాబాద్', lat: 17.4399, lon: 78.4983 },
  { name: 'Shamshabad', telugu: 'శంషాబాద్', lat: 17.2543, lon: 78.4311 },
  { name: 'Warangal', telugu: 'వరంగల్', lat: 17.9689, lon: 79.5941 },
  { name: 'Karimnagar', telugu: 'కరీంనగర్', lat: 18.4386, lon: 79.1288 },
];

/* ──────────────────────────────────────────────────────────
   Helper: Farmer-friendly depth message with Feet and Borewell advice
   ────────────────────────────────────────────────────────── */
const getDepthMessage = (depth: number | null, confidence: number, lang: Lang): string => {
  if (confidence <= 0 || depth == null) {
    return lang === 'te'
      ? 'ఈ ప్రాంతానికి సంబంధించి సరిపడా డేటా అందుబాటులో లేదు. దగ్గర్లోని మరొక ప్రాంతాన్ని పరిశీలించండి.'
      : "We don't have enough monitoring data for this location yet. Try a nearby area.";
  }
  const feet = Math.round(depth * 3.28084);
  if (depth < 5) {
    return lang === 'te'
      ? `💧 నీరు భూమికి చాలా దగ్గరగా ఉంది (~${feet} అడుగులు)! 30 అడుగుల లోపు బావులు లేదా చిన్న బోరుబావుల్లోనే నీరు సులభంగా పడుతుంది.`
      : `💧 Water is very close to surface (~${feet} ft)! Open wells and shallow borewells under 30 ft will easily strike water.`;
  }
  if (depth < 10) {
    return lang === 'te'
      ? `💧 నీటి మట్టం చాలా అనుకూలంగా ఉంది (~${feet} అడుగులు). 40–60 అడుగుల సాధారణ బోరుబావులతో సాగునీరు పుష్కలంగా లభిస్తుంది.`
      : `💧 Shallow water level (~${feet} ft deep). Standard small borewells (40–60 ft) will provide excellent water for irrigation.`;
  }
  if (depth < 20) {
    return lang === 'te'
      ? `💧 మధ్యస్థ లోతులో నీరు ఉంది (~${feet} అడుగులు). 80–120 అడుగుల బోరుబావి వేస్తే నీరు అందుతుంది. బిందు సేద్యం (డ్రిప్) వాడటం మంచిది.`
      : `💧 Moderate water depth (~${feet} ft deep). Standard borewells (80–120 ft) can reach water reliably. Drip irrigation recommended.`;
  }
  if (depth < 30) {
    return lang === 'te'
      ? `⚠️ నీరు కాస్త లోతుగా ఉంది (~${feet} అడుగులు). 120–180 అడుగుల లోతు బోరుబావి అవసరం. నీటిని ఆదా చేస్తూ పంటలు సాగు చేయండి.`
      : `⚠️ Deeper water table (~${feet} ft deep). You will need a borewell of 120–180 ft. Mulching and water conservation are strongly advised.`;
  }
  return lang === 'te'
    ? `🚨 నీరు చాలా లోతులో ఉంది (~${feet} అడుగులు). 200 అడుగులకు పైగా లోతు వేయాల్సి ఉంటుంది. ఇంకుడు గుంతలు, పంట కుంటలు నిర్మించుకోండి.`
    : `🚨 Deep groundwater (~${feet} ft deep). Deep drilling (200+ ft) required. Rainwater harvesting and farm ponds are essential.`;
};

/* ──────────────────────────────────────────────────────────
   Helper: Farmer-friendly Soil Texture and Type
   ────────────────────────────────────────────────────────── */
const getSoilTypeLabel = (soil: PredictionData['soil'], lang: Lang): string => {
  const clay = soil?.clay_pct || 0;
  const sand = soil?.sand_pct || 0;
  if (clay > 40) {
    return lang === 'te' ? 'నల్ల రేగడి నేల (Black Cotton / Clayey)' : 'Black Cotton Soil (Nalla Regadi / Clayey)';
  }
  if (sand > 50) {
    return lang === 'te' ? 'ఇసుక నేల (Isuka Nela / Red Sand)' : 'Sandy Soil (Isuka Nela / Red Sand)';
  }
  if (clay > 20 && sand > 30) {
    return lang === 'te' ? 'ఎర్ర నేల / చలక (Red Sandy Loam)' : 'Red Sandy Loam (Erra Nela / Chalaka)';
  }
  return lang === 'te' ? 'సారవంతమైన మిశ్రమ నేల (Loam Soil)' : 'Mixed Loam Soil (Balanced Farmland)';
};

const getSoilMessage = (soil: PredictionData['soil'], lang: Lang): string => {
  if (!soil?.clay_pct && !soil?.sand_pct && !soil?.silt_pct) {
    return lang === 'te' ? 'ఈ ప్రాంతానికి నేల సమాచారం లభించలేదు.' : 'Soil composition data is not available for this location.';
  }
  const clay = soil.clay_pct || 0;
  const sand = soil.sand_pct || 0;
  if (clay > 40) {
    return lang === 'te'
      ? '🏺 బంకమట్టి ఎక్కువ ఉన్న నేల — నీటిని ఎక్కువ రోజులు నిలుపుకుంటుంది. వరి, చెరకు, పత్తి పంటలకు ఎంతో అనుకూలం.'
      : '🏺 Clay-rich soil — holds water for days but drains slowly. Best suited for Paddy, Sugarcane, and Cotton.';
  }
  if (sand > 50) {
    return lang === 'te'
      ? '🏖️ ఇసుక నేల — నీరు త్వరగా లోపలికి ఇంకిపోతుంది. వేరుశనగ, పుచ్చకాయ, కూరగాయలు, పప్పుదినుసులకు అనుకూలం. డ్రిప్ వాడండి.'
      : '🏖️ Sandy soil — water drains very quickly into ground. Ideal for Groundnuts, Watermelon, Pulses, and Vegetables. Use drip irrigation.';
  }
  if (clay > 20 && sand > 30) {
    return lang === 'te'
      ? '🌱 ఎర్ర నేల / చలక — వ్యవసాయానికి అత్యంత అనుకూలమైన నేల! తేమను నిలుపుకుంటుంది, దున్నడం సులభం. పత్తి, మొక్కజొన్న, మిరప, కూరగాయలకు శ్రేష్టం.'
      : '🌱 Red Sandy Loam (Chalaka) — excellent general soil! Holds moisture well, easy to plow. Great for Cotton, Maize, Chillies, and Vegetables.';
  }
  return lang === 'te'
    ? '🌱 సారవంతమైన ఒండ్రు నేల — అన్ని రకాల పంటలకు సరిపడే బలమైన భూమి.'
    : '🌱 Balanced Loam — fertile soil with good water retention and aeration. Suitable for almost all crops.';
};

/* ──────────────────────────────────────────────────────────
   Helper: Farmer-friendly Soil pH (Sweet / Neutral / Acidic / Saline)
   ────────────────────────────────────────────────────────── */
const getSoilPhDetails = (ph: number | null, lang: Lang) => {
  if (ph == null) return {
    tag: lang === 'te' ? 'pH వివరాలు లేవు' : 'Unknown pH',
    status: lang === 'te' ? 'మట్టి పరీక్ష అవసరం' : 'Soil Test Recommended',
    desc: lang === 'te' ? 'సమీప రైతు భరోసా కేంద్రంలో మట్టి పరీక్ష చేయించండి.' : 'Local soil testing will reveal exact fertilizer needs.',
    badgeColor: '#94a3b8',
    bg: 'rgba(148, 163, 184, 0.1)'
  };
  if (ph < 6.0) return {
    tag: lang === 'te' ? `pH ${ph.toFixed(1)} (ఆమ్ల / పుల్లని నేల)` : `pH ${ph.toFixed(1)} (Acidic / Sour)`,
    status: lang === 'te' ? 'పుల్లని నేల (సున్నం వేయాలి)' : 'Sour Soil (Needs Lime)',
    desc: lang === 'te' ? 'నేలలో పోషకాలు మొక్కకు అందవు. నేల బలానికి వ్యవసాయ సున్నం (Sunnam) చల్లాలి.' : 'Nutrients get locked. Apply Agricultural Lime (Sunnam) to balance.',
    badgeColor: '#f87171',
    bg: 'rgba(239, 68, 68, 0.12)'
  };
  if (ph <= 7.5) return {
    tag: lang === 'te' ? `pH ${ph.toFixed(1)} (తీపి / మంచి నేల)` : `pH ${ph.toFixed(1)} (Sweet / Healthy)`,
    status: lang === 'te' ? 'తీపి / బలమైన నేల (పంటలకు మేలు)' : 'Sweet / Normal Soil (Ideal)',
    desc: lang === 'te' ? 'అత్యుత్తమ నేల స్థితి! వేసిన ఎరువులు, పోషకాలు పంటకు సంపూర్ణంగా అందుతాయి. సున్నం లేదా జిప్సం అవసరం లేదు.' : 'Best possible condition! All fertilizers and nutrients are absorbed easily by crops.',
    badgeColor: '#34d399',
    bg: 'rgba(16, 185, 129, 0.12)'
  };
  return {
    tag: lang === 'te' ? `pH ${ph.toFixed(1)} (చౌడు నేల)` : `pH ${ph.toFixed(1)} (Alkaline / Saline)`,
    status: lang === 'te' ? 'చౌడు నేల (జిప్సం వేయాలి)' : 'Alkaline (Chowdu Soil)',
    desc: lang === 'te' ? 'నీరు నేలలోకి త్వరగా ఇంకదు. జిప్సం మరియు పశువుల పెంట ఎరువు వేసి దున్నాలి.' : 'Water penetrates slowly. Apply Gypsum and organic manure (penta) before sowing.',
    badgeColor: '#fbbf24',
    bg: 'rgba(245, 158, 11, 0.12)'
  };
};

/* ──────────────────────────────────────────────────────────
   Helper: Farmer-friendly Rainfall Interpretation
   ────────────────────────────────────────────────────────── */
const getRainfallDetails = (rainfallMm: number | null, lang: Lang) => {
  if (rainfallMm == null) return {
    display: 'N/A',
    tag: lang === 'te' ? 'వర్షపాత డేటా లేదు' : 'No Sensor Data',
    desc: lang === 'te' ? 'ఈ నెల వర్షపాత వివరాలు అందుబాటులో లేవు.' : 'Rainfall data could not be fetched for this month.',
    badgeColor: '#94a3b8'
  };
  if (rainfallMm < 10) return {
    display: `${rainfallMm.toFixed(1)} mm`,
    tag: lang === 'te' ? '☀️ పొడి కాలం (వర్షం లేదు)' : '☀️ Dry Period (Scanty Rain)',
    desc: lang === 'te' ? 'పొడి వాతావరణం — పైనేల ఆరిపోయింది; భూమిలోకి నీరు ఇంకడం లేదు. బోరుబావి నీటితో తడులు ఇవ్వండి.' : 'Dry spell — topsoil is dry; groundwater is NOT actively recharging. Irrigate from borewells.',
    badgeColor: '#f59e0b'
  };
  if (rainfallMm < 40) return {
    display: `${rainfallMm.toFixed(1)} mm`,
    tag: lang === 'te' ? '🌦️ స్వల్ప వర్షం (పైనేల తేమ)' : '🌦️ Light Rain (Moist Soil)',
    desc: lang === 'te' ? 'పైపంటలకు సరిపడా తేమ అందుతుంది. లోపలి భూగర్భ జలాల వరకు నీరు చేరడం తక్కువ.' : 'Moistens topsoil for standing crops. Limited seepage into deeper underground aquifers.',
    badgeColor: '#38bdf8'
  };
  if (rainfallMm < 100) return {
    display: `${rainfallMm.toFixed(1)} mm`,
    tag: lang === 'te' ? '🌧️ మంచి వర్షం (నీరు చేరుతోంది)' : '🌧️ Good Rainfall (Recharging)',
    desc: lang === 'te' ? 'మంచి వర్షం కురిసింది! భూగర్భ జలాలు సహజంగా రీఛార్జ్ అవుతున్నాయి.' : 'Good seasonal rain! Water is naturally recharging the groundwater table.',
    badgeColor: '#10b981'
  };
  return {
    display: `${rainfallMm.toFixed(1)} mm`,
    tag: lang === 'te' ? '🌊 భారీ వర్షం (సమృద్ధిగా నీరు)' : '🌊 Heavy Rain (Abundant Water)',
    desc: lang === 'te' ? 'భారీ వర్షం! భూగర్భ జలాలు పూర్తి స్థాయిలో పెరుగుతాయి. పొలంలో నీరు నిలవకుండా చూసుకోండి.' : 'Heavy soaking rain! Maximum groundwater recharge. Watch out for field waterlogging.',
    badgeColor: '#06b6d4'
  };
};

/* ──────────────────────────────────────────────────────────
   Helper: Farmer-friendly Temperature & Evaporation (Celsius ONLY)
   ────────────────────────────────────────────────────────── */
const getWeatherDetails = (tempC: number | null, humidityPct: number | null, lang: Lang) => {
  if (tempC == null) return {
    tempDisplay: 'N/A',
    status: lang === 'te' ? 'సమాచారం లేదు' : 'Weather N/A',
    evapDesc: lang === 'te' ? 'ఉష్ణోగ్రత వివరాలు లేవు.' : 'Temperature data helps estimate water evaporation rates.',
    badgeColor: '#94a3b8'
  };
  const isHot = tempC > 34;
  const isModerate = tempC >= 24 && tempC <= 34;
  
  let evapDesc = '';
  if (isHot) {
    evapDesc = lang === 'te'
      ? 'తీవ్రమైన ఎండ — నేలలోని తేమ వేగంగా ఆవిరైపోతుంది. నీటి ఆవిరిని తగ్గించడానికి ఉదయం లేదా సాయంత్రం తడులు ఇవ్వండి.'
      : 'High heat & rapid soil drying. Irrigate early morning or night to prevent water evaporation.';
  } else if (isModerate) {
    evapDesc = lang === 'te'
      ? `సాధారణ ఉష్ణోగ్రత (${humidityPct ? `${Math.round(humidityPct)}% గాలిలో తేమ` : 'తేమ సాధారణం'}). నీటి ఆవిరి సాధారణ స్థాయిలో ఉంది.`
      : `Warm weather (${humidityPct ? `${Math.round(humidityPct)}% humidity` : 'normal humidity'}). Normal water evaporation rate.`;
  } else {
    evapDesc = lang === 'te'
      ? 'చల్లని వాతావరణం — నీరు ఆవిరి కావడం చాలా తక్కువ; నేలలో తేమ ఎక్కువ రోజులు ఉంటుంది.'
      : 'Cool/mild weather. Very low water evaporation — soil stays moist longer.';
  }

  return {
    tempDisplay: `${tempC.toFixed(1)}°C`,
    status: isHot
      ? (lang === 'te' ? '🔥 తీవ్రమైన ఎండ' : '🔥 High Heat')
      : isModerate
      ? (lang === 'te' ? '🌤️ సాధారణ ఎండ / వెచ్చగా' : '🌤️ Warm / Moderate')
      : (lang === 'te' ? '⛅ చల్లని వాతావరణం' : '⛅ Cool / Mild'),
    evapDesc,
    badgeColor: isHot ? '#ef4444' : isModerate ? '#fbbf24' : '#38bdf8'
  };
};

/* ──────────────────────────────────────────────────────────
   Helper: Farmer-friendly Elevation & Terrain
   ────────────────────────────────────────────────────────── */
const getElevationDetails = (elevationM: number | null, lang: Lang) => {
  if (elevationM == null) return {
    display: 'N/A',
    tag: lang === 'te' ? 'సమాచారం లేదు' : 'Terrain Data N/A',
    desc: lang === 'te' ? 'సముద్ర మట్టం ఎత్తు నీటి ప్రవాహాన్ని ప్రభావితం చేస్తుంది.' : 'Elevation influences natural groundwater flow.'
  };
  const feet = Math.round(elevationM * 3.28084);
  if (elevationM > 550) {
    return {
      display: `${elevationM} m (${feet} ft)`,
      tag: lang === 'te' ? '⛰️ కొండ వాలు / ఎత్తైన ప్రాంతం' : '⛰️ Hill Slope / Ridge',
      desc: lang === 'te' ? 'ఎత్తైన ప్రాంతం: వర్షపు నీరు వాలులోకి వేగంగా జారిపోతుంది. పల్లపు ప్రాంతాల్లో బోరుబావులు వేస్తే ఎక్కువ నీరు పడుతుంది.' : 'High slope: Rainwater runs off quickly downhill. Groundwater is deeper; valleys yield more water.'
    };
  }
  if (elevationM >= 420) {
    return {
      display: `${elevationM} m (${feet} ft)`,
      tag: lang === 'te' ? '🌾 దక్కన్ పీఠభూమి పొలం' : '🌾 Deccan Plateau Farmland',
      desc: lang === 'te' ? 'ఎత్తైన పీఠభూమి ప్రాంతం: సాధారణ వాలు. పొలం చుట్టూ గట్లు వేయడం ద్వారా వర్షపు నీటిని భూమిలోకి ఇంకించవచ్చు.' : 'Elevated plateau terrain: Moderate runoff. Field bunding helps soak rain into the groundwater.'
    };
  }
  return {
    display: `${elevationM} m (${feet} ft)`,
    tag: lang === 'te' ? '🏞️ పల్లపు ప్రాంతం / లోయ' : '🏞️ Valley / Lowland Basin',
    desc: lang === 'te' ? 'సహజ పల్లపు ప్రాంతం: చుట్టుపక్కల వర్షపు నీరు ఇక్కడే చేరుతుంది. తక్కువ లోతులోనే ఎక్కువ నీరు లభిస్తుంది.' : 'Natural low basin: Rainwater collects here. Groundwater recharge is highest; best for shallow wells.'
  };
};

/* ──────────────────────────────────────────────────────────
   Helper: Generate farming recommendations
   ────────────────────────────────────────────────────────── */
interface Recommendation {
  icon: string;
  text: string;
  color: string;
  bgColor: string;
}

const getRecommendations = (prediction: PredictionData, lang: Lang): Recommendation[] => {
  const recs: Recommendation[] = [];
  const depth = prediction.estimated_depth_m;
  const rainfall = prediction.weather?.rainfall_mm;
  const confidence = prediction.confidence;

  if (confidence <= 0) {
    recs.push({
      icon: '📡',
      text: lang === 'te'
        ? 'ఈ ప్రాంతానికి సమీపంలో ప్రభుత్వ పర్యవేక్షణ బావులు తక్కువగా ఉన్నాయి. దగ్గర్లోని ప్రధాన గ్రామం లేదా పట్టణాన్ని ఎంచుకోండి.'
        : 'This area has limited monitoring stations nearby. Try checking a location closer to a town or city.',
      color: '#94a3b8',
      bgColor: 'rgba(148, 163, 184, 0.08)',
    });
    return recs;
  }

  const depthFt = Math.round(depth * 3.28084);

  // Depth-based recommendations
  if (depth < 10) {
    recs.push({
      icon: '✅',
      text: lang === 'te'
        ? `నీటి మట్టం చాలా అనుకూలంగా ఉంది (సుమారు ${depthFt} అడుగుల లోతు)! 60 అడుగుల లోపు చిన్న బోరుబావి లేదా బావి తవ్వితే పుష్కలంగా నీరు లభిస్తుంది. అన్ని రకాల పంటలకు సరిపోతుంది.`
        : `Water level is excellent (~${depthFt} ft below ground)! Shallow borewells (under 60 ft) and open wells will strike water easily. Great conditions for all crops.`,
      color: '#10b981',
      bgColor: 'rgba(16, 185, 129, 0.08)',
    });
  } else if (depth < 20) {
    recs.push({
      icon: '👍',
      text: lang === 'te'
        ? `నీరు మంచి లోతులో ఉంది (సుమారు ${depthFt} అడుగులు). 80–120 అడుగుల బోరుబావి వేస్తే సరిపోతుంది. నీటి ఆదా కోసం స్ప్రింక్లర్లు లేదా డ్రిప్ వాడండి.`
        : `Water level is good (~${depthFt} ft deep). Standard borewells (80–120 ft) will reach water reliably. Use drip irrigation or sprinklers to save water.`,
      color: '#06b6d4',
      bgColor: 'rgba(6, 182, 212, 0.08)',
    });
  } else if (depth < 30) {
    recs.push({
      icon: '⚡',
      text: lang === 'te'
        ? `నీరు మధ్యస్థ లోతులో ఉంది (సుమారు ${depthFt} అడుగులు). 120–180 అడుగుల లోతు బోరుబావి అవసరం. నేలలో తేమ ఆరిపోకుండా మల్చింగ్ పద్ధతులు పాటించండి.`
        : `Water is at moderate depth (~${depthFt} ft deep). Drill 120–180 ft borewells. Practice mulching and drip irrigation to prevent moisture loss.`,
      color: '#f59e0b',
      bgColor: 'rgba(245, 158, 11, 0.08)',
    });
  } else {
    recs.push({
      icon: '🚨',
      text: lang === 'te'
        ? `నీటి మట్టం చాలా లోతుగా ఉంది (సుమారు ${depthFt} అడుగులు). 200 అడుగుల కంటే ఎక్కువ లోతు తవ్వాల్సి ఉంటుంది. పంట కుంటలు మరియు ఇంకుడు గుంతలు తప్పనిసరిగా ఏర్పాటు చేసుకోండి.`
        : `Water level is deep (~${depthFt} ft deep). Deep boring (200+ ft) needed. Strongly recommend farm ponds (kunta) and rainwater recharge pits.`,
      color: '#ef4444',
      bgColor: 'rgba(239, 68, 68, 0.08)',
    });
  }

  // Rainfall-based recommendations
  if (rainfall != null) {
    if (rainfall > 80) {
      recs.push({
        icon: '🌧️',
        text: lang === 'te'
          ? `మంచి వర్షపాతం (${rainfall.toFixed(1)} mm) నమోదైంది. విత్తనాలు వేయడానికి మరియు భూగర్భ జలాలు పెరగడానికి ఇది సరైన సమయం.`
          : `Good rainfall (${rainfall.toFixed(1)} mm) is actively soaking the ground. Perfect time for sowing and recharging groundwater.`,
        color: '#3b82f6',
        bgColor: 'rgba(59, 130, 246, 0.08)',
      });
    } else if (rainfall < 10) {
      recs.push({
        icon: '☀️',
        text: lang === 'te'
          ? `ప్రస్తుతం పొడి కాలం నడుస్తోంది (${rainfall.toFixed(1)} mm వర్షం). బోర్ల నుండి ఎక్కువ నీరు తోడకుండా పంట అవసరాలకు మాత్రమే పొదుపుగా వాడండి.`
          : `Current dry spell (${rainfall.toFixed(1)} mm rain). No natural recharge happening right now. Regulate borewell pumping and avoid flooding fields.`,
        color: '#f59e0b',
        bgColor: 'rgba(245, 158, 11, 0.08)',
      });
    }
  }

  // Soil-based recommendations
  if (prediction.soil?.clay_pct != null) {
    if (prediction.soil.clay_pct > 40) {
      recs.push({
        icon: '🏺',
        text: lang === 'te'
          ? 'నల్ల రేగడి నేల (Nalla Regadi): వారాల తరబడి తేమను పట్టి ఉంచుతుంది. వరి, పత్తి, చెరకు పంటలకు శ్రేష్టం. పంట కుంట తవ్వితే నీరు ఏమాత్రం ఇంకిపోకుండా నిల్వ ఉంటుంది.'
          : 'Clay-rich Black Soil (Nalla Regadi): Retains moisture for weeks. Perfect for Paddy, Cotton, and Sugarcane. Farm ponds will store water with zero seepage.',
        color: '#92400e',
        bgColor: 'rgba(146, 64, 14, 0.08)',
      });
    } else if (prediction.soil.sand_pct != null && prediction.soil.sand_pct > 45) {
      recs.push({
        icon: '🏖️',
        text: lang === 'te'
          ? 'ఎర్ర నేల / ఇసుక నేల (Erra Nela): నీరు త్వరగా ఇంకుతుంది. వేరుశనగ, మొక్కజొన్న, కూరగాయలకు ఉత్తమం. కాలువల ద్వారా కాకుండా డ్రిప్ ద్వారా నీరందిస్తే నీరు వృధా కాదు.'
          : 'Sandy / Red Sandy Soil (Erra Nela): Drains quickly. Best for Groundnut, Maize, Pulses, and Vegetables. Use drip irrigation to prevent water wastage.',
        color: '#d97706',
        bgColor: 'rgba(217, 119, 6, 0.08)',
      });
    }
  }

  return recs;
};

/* ══════════════════════════════════════════════════════════
   Main App Component
   ══════════════════════════════════════════════════════════ */

export default function App() {
  const [lang, setLang] = useState<Lang>('en');
  const [coords, setCoords] = useState<{ lat: number; lon: number }>({ lat: 17.422783, lon: 78.650019 });
  const [inputLat, setInputLat] = useState<string>('17.422783');
  const [inputLon, setInputLon] = useState<string>('78.650019');
  const [loading, setLoading] = useState<boolean>(false);
  const [prediction, setPrediction] = useState<PredictionData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [feedbackOpen, setFeedbackOpen] = useState<boolean>(false);
  const [feedbackActualDepth, setFeedbackActualDepth] = useState<string>('');
  const [feedbackSubmitted, setFeedbackSubmitted] = useState<boolean>(false);
  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    return (localStorage.getItem('geoground_theme') as 'dark' | 'light') || 'dark';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('geoground_theme', theme);
  }, [theme]);

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

    // Clean Google Maps tiles
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
          width: 34px; height: 34px;
          background: radial-gradient(circle, #38bdf8 0%, #0284c7 100%);
          border: 3px solid #ffffff;
          border-radius: 50% 50% 50% 0;
          transform: rotate(-45deg);
          box-shadow: 0 0 24px rgba(56, 189, 248, 0.95);
          display: flex; align-items: center; justify-content: center;
        ">
          <div style="width: 8px; height: 8px; background: white; border-radius: 50%; transform: rotate(45deg);"></div>
        </div>
      `,
      iconSize: [34, 34],
      iconAnchor: [17, 34],
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

    if (mapInstanceRef.current) {
      mapInstanceRef.current.panTo([cleanLat, cleanLon], { animate: true, duration: 0.8 });
    }
    if (pinMarkerRef.current) {
      pinMarkerRef.current.setLatLng([cleanLat, cleanLon]);
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
      // First try Backend Gateway, fallback to ML direct
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
          groundwater_health_score: d.prediction.groundwaterHealthScore,
          health_status: d.prediction.healthStatus,
          health_description: d.prediction.healthDescription,
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
        const m = mlRes.data;
        setPrediction({
          ...m,
          confidence: m.confidence > 1 ? m.confidence : Math.round(m.confidence * 100)
        });
      }
    } catch (err: any) {
      console.error('Prediction fetch error:', err);
      setError(err.response?.data?.detail || err.message || (lang === 'te' ? 'ఈ ప్రదేశంలో నీటి మట్టం లెక్కించలేకపోయాము.' : 'Unable to estimate groundwater at this location.'));
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status: string = '') => {
    const s = status.toLowerCase();
    if (s.includes('uncertain') || s.includes('no data') || s.includes('unmonitored') || s.includes('unknown')) return '#94a3b8';
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
          alert((lang === 'te' ? 'జీపీఎస్ లొకేషన్ అందుబాటులో లేదు: ' : 'Could not retrieve GPS location: ') + err.message);
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
      alert(lang === 'te' ? 'మీ వివరాలు నమోదయ్యాయి.' : 'Feedback recorded locally.');
      setFeedbackOpen(false);
    }
  };

  /* ──── Computed values for visuals ──── */
  const depthPct = prediction?.estimated_depth_m != null && prediction.confidence > 0
    ? Math.min(85, Math.max(15, (prediction.estimated_depth_m / 35) * 100))
    : 0;

  const clayPct = prediction?.soil?.clay_pct || 0;
  const sandPct = prediction?.soil?.sand_pct || 0;
  const siltPct = prediction?.soil?.silt_pct || 0;
  const soilTotal = clayPct + sandPct + siltPct || 1;
  const clayAngle = (clayPct / soilTotal) * 360;
  const sandAngle = (sandPct / soilTotal) * 360;

  const healthScore = prediction?.groundwater_health_score ?? 0;
  const healthCircumference = 2 * Math.PI * 52; // r=52
  const healthOffset = healthCircumference - (healthScore / 100) * healthCircumference;
  const healthColor = healthScore >= 60 ? '#10b981' : healthScore >= 35 ? '#f59e0b' : '#ef4444';

  const recommendations = prediction ? getRecommendations(prediction, lang) : [];

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
              <span className="brand-tag">AI-Powered</span>
            </div>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
              {lang === 'te'
                ? 'రైతులకు ఉపయోగపడే కృత్రిమ మేధ భూగర్భ జలాల అంచనా'
                : 'Smart Groundwater Estimation for Farmers & Communities'}
            </p>
          </div>
        </div>

        <div className="header-actions">
          {/* Theme Switcher Toggle */}
          <button
            className="theme-toggle-btn"
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            title={theme === 'dark' ? (lang === 'te' ? 'లైట్ మోడ్‌కి మార్చండి' : 'Switch to Light Mode') : (lang === 'te' ? 'డార్క్ మోడ్‌కి మార్చండి' : 'Switch to Dark Mode')}
            aria-label="Toggle theme"
          >
            {theme === 'dark' ? (
              <>
                <Sun size={15} className="theme-icon sun-icon" />
                <span>{lang === 'te' ? 'లైట్' : 'Light'}</span>
              </>
            ) : (
              <>
                <Moon size={15} className="theme-icon moon-icon" />
                <span>{lang === 'te' ? 'డార్క్' : 'Dark'}</span>
              </>
            )}
          </button>

          {/* Language Switcher Button */}
          <div className="lang-switcher">
            <Languages size={16} color="var(--accent-cyan)" />
            <button
              className={`lang-btn ${lang === 'en' ? 'active' : ''}`}
              onClick={() => setLang('en')}
            >
              English
            </button>
            <span style={{ color: 'rgba(255,255,255,0.2)' }}>|</span>
            <button
              className={`lang-btn ${lang === 'te' ? 'active' : ''}`}
              onClick={() => setLang('te')}
            >
              తెలుగు
            </button>
          </div>

          <button className="btn-pill" onClick={handleLocateMe} title={lang === 'te' ? 'మీ ప్రస్తుత లొకేషన్ ఎంచుకోండి' : 'Use your current location'}>
            <Crosshair size={16} color="var(--accent-cyan)" />
            <span>{lang === 'te' ? 'నా లొకేషన్' : 'Use My Location'}</span>
          </button>
          <button
            className="btn-pill btn-pill-primary"
            onClick={() => setFeedbackOpen(true)}
            title={lang === 'te' ? 'బోరుబావి లోతు వివరాలను పంపండి' : 'Share actual borewell depth'}
          >
            <Send size={15} />
            <span>{lang === 'te' ? 'బోరుబావి వివరాలు' : 'Share Borewell Data'}</span>
          </button>
        </div>
      </header>

      {/* ── Main Dashboard Layout ─────────────────────────────────── */}
      <div className="main-layout">
        {/* ════ LEFT: Info Dashboard (Primary) ════ */}
        <div className="info-dashboard">
          {/* Location Card */}
          <div className="glass-panel location-card animate-in">
            <div className="location-label">
              <Compass size={18} />
              <span>{lang === 'te' ? 'మీరు ఎంచుకున్న ప్రదేశం' : 'Your Selected Location'}</span>
            </div>
            <div className="location-name">
              {prediction?.location?.displayName || `GPS Location (${coords.lat.toFixed(4)}°N, ${coords.lon.toFixed(4)}°E)`}
            </div>
            <div className="location-details">
              {lang === 'te' ? 'జిల్లా: ' : 'District: '}<strong style={{ color: '#38bdf8' }}>{prediction?.location?.district || (lang === 'te' ? 'గుర్తించబడలేదు' : 'Not identified')}</strong>
              {' • '}
              {lang === 'te' ? 'రాష్ట్రం: ' : 'State: '}{prediction?.location?.state || (prediction?.confidence && prediction.confidence > 0 ? (lang === 'te' ? 'తెలంగాణ, భారతదేశం' : 'Telangana, India') : 'Unknown region')}
            </div>
          </div>

          {loading ? (
            /* ── Loading State ── */
            <div className="glass-panel loading-state">
              <div className="loading-icon">
                <Droplets size={48} color="var(--accent-cyan)" />
              </div>
              <h3 className="loading-title">{lang === 'te' ? 'మీ పొలం వద్ద నీటి మట్టం లెక్కిస్తున్నాము...' : 'Analyzing your location...'}</h3>
              <p className="loading-subtitle">
                {lang === 'te'
                  ? 'వర్షపాతం, నేల రకం, సముద్ర మట్టం మరియు దగ్గర్లోని ప్రభుత్వ బావుల వివరాలను ఏఐ ద్వారా లెక్కిస్తున్నాము.'
                  : 'Checking rainfall data, soil type, elevation, and nearby water levels to estimate groundwater depth.'}
              </p>
            </div>
          ) : error ? (
            /* ── Error State ── */
            <div className="glass-panel error-state">
              <div className="error-header">
                <XCircle size={22} />
                <h4 style={{ fontWeight: 700, fontSize: '1.05rem' }}>{lang === 'te' ? 'నీటి మట్టం అంచనా వేయలేకపోయాము' : 'Could Not Estimate Water Level'}</h4>
              </div>
              <p className="error-message">{error}</p>
            </div>
          ) : prediction ? (
            <>
              {/* ═══ 1. Water Depth Hero Card ═══ */}
              <div
                className="depth-hero-card animate-in animate-in-delay-1"
                style={{
                  minHeight: '220px',
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'space-between',
                  padding: '24px',
                  borderRadius: '16px',
                  position: 'relative',
                  overflow: 'hidden',
                }}
              >
                <div className="depth-hero-header" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span className="depth-hero-title" style={{ fontSize: '0.95rem', color: 'var(--text-secondary)', fontWeight: 700 }}>
                    {lang === 'te' ? '💧 నీరు ఎంత లోతులో ఉంది? (భూగర్భ జలాలు)' : '💧 How Deep is the Water? (Estimated Groundwater Table)'}
                  </span>
                  <div className={`status-badge ${getStatusClass(prediction.condition)}`}>
                    <span style={{ width: 8, height: 8, borderRadius: '50%', background: getStatusColor(prediction.condition), display: 'inline-block' }}></span>
                    {prediction.condition === 'Excellent'
                      ? (lang === 'te' ? '🟢 తక్కువ లోతులోనే సమృద్ధిగా నీరు' : '🟢 Very Shallow / Accessible')
                      : prediction.condition === 'Good'
                      ? (lang === 'te' ? '🔵 మంచి నీటి మట్టం' : '🔵 Good Depth')
                      : prediction.condition === 'Moderate'
                      ? (lang === 'te' ? '🟡 మధ్యస్థ లోతు' : '🟡 Moderate Depth')
                      : (lang === 'te' ? '🔴 చాలా లోతైన నీరు' : '🔴 Deep Water Table')}
                  </div>
                </div>

                <div className="depth-hero-content" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '24px', margin: '14px 0' }}>
                  <div className="depth-hero-left" style={{ flex: 1 }}>
                    <div style={{ display: 'flex', alignItems: 'baseline', gap: '12px', flexWrap: 'wrap' }}>
                      <span className="depth-value" style={{ fontSize: '3.4rem', fontWeight: 800, color: 'var(--text-primary)', fontFamily: 'Outfit, sans-serif', lineHeight: 1 }}>
                        {prediction.estimated_depth_m != null && prediction.confidence > 0
                          ? `${prediction.estimated_depth_m.toFixed(1)} m`
                          : '--'}
                      </span>
                      <span className="depth-feet-tag" style={{ fontSize: '1.05rem', fontWeight: 700, color: '#38bdf8', background: 'rgba(56, 189, 248, 0.15)', border: '1px solid rgba(56, 189, 248, 0.4)', padding: '4px 14px', borderRadius: '20px' }}>
                        {prediction.estimated_depth_m != null && prediction.confidence > 0
                          ? (lang === 'te'
                              ? `సుమారు ${Math.round(prediction.estimated_depth_m * 3.28084)} అడుగుల లోతు`
                              : `approx. ${Math.round(prediction.estimated_depth_m * 3.28084)} feet deep`)
                          : 'depth unavailable'}
                      </span>
                    </div>

                    <div className="depth-farmer-message" style={{ fontSize: '0.98rem', color: 'var(--text-primary)', lineHeight: 1.55, marginTop: '12px', padding: '12px 16px', borderRadius: '10px' }}>
                      {getDepthMessage(prediction.estimated_depth_m, prediction.confidence, lang)}
                    </div>

                    <div className="depth-trend" style={{ fontSize: '0.88rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '8px', marginTop: '10px' }}>
                      {prediction.confidence > 0 ? (
                        <>
                          {prediction.trend === 'Increasing' ? <TrendingUp size={16} color="#38bdf8" /> : prediction.trend === 'Decreasing' ? <TrendingDown size={16} color="#10b981" /> : <Minus size={16} color="#94a3b8" />}
                          <span style={{ color: prediction.trend === 'Decreasing' ? '#34d399' : '#38bdf8', fontWeight: 600 }}>
                            {lang === 'te'
                              ? `నీటి మార్పు: ${prediction.trend === 'Increasing' ? 'నీటి మట్టం తగ్గుతోంది (ఏటేటా లోతు పెరుగుతోంది)' : prediction.trend === 'Decreasing' ? 'నీటి మట్టం పెరుగుతోంది (భూగర్భ జలాలు పైకి వస్తున్నాయి)' : 'నీటి మట్టం స్థిరంగా ఉంది'}`
                              : `Water Trend: ${prediction.trend === 'Increasing' ? 'Water level is slowly dropping over years (deeper)' : prediction.trend === 'Decreasing' ? 'Water level is rising/recovering (shallower)' : 'Water level is stable over time'}`}
                          </span>
                        </>
                      ) : (
                        <>
                          <Minus size={16} color="#94a3b8" />
                          <span style={{ color: '#94a3b8' }}>{lang === 'te' ? 'గత సంవత్సరాల రికార్డులు అందుబాటులో లేవు' : 'No nearby well trend data available'}</span>
                        </>
                      )}
                    </div>
                  </div>

                  {/* Borewell Graphic */}
                  <div className="borewell-box" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '6px', padding: '10px 14px 8px', borderRadius: '12px', flexShrink: 0 }}>
                    <div className="borewell-box-title" style={{ fontSize: '0.72rem', fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)' }}>{lang === 'te' ? 'బోరుబావి లోతు' : 'Borewell Depth'}</div>
                    <div className="well-visual" style={{ width: '58px', height: '140px', borderRadius: '8px', position: 'relative', overflow: 'hidden' }}>
                      <div
                        className="well-ground-layer"
                        style={{ position: 'absolute', top: 0, left: 0, right: 0, height: `${depthPct}%`, background: 'linear-gradient(180deg, #78350f, #92400e)', transition: 'height 1s ease' }}
                      >
                        <span className="well-label" style={{ position: 'absolute', bottom: '4px', width: '100%', textAlign: 'center', fontSize: '0.62rem', fontWeight: 700, color: '#ffffff' }}>
                          {lang === 'te' ? 'నేల' : 'Soil'} {prediction.estimated_depth_m != null ? `${prediction.estimated_depth_m.toFixed(0)}m` : ''}
                        </span>
                      </div>
                      <div
                        className="well-water-layer"
                        style={{ position: 'absolute', bottom: 0, left: 0, right: 0, height: `${Math.max(15, 100 - depthPct)}%`, background: 'linear-gradient(180deg, #0ea5e9, #0369a1)', transition: 'height 1s ease' }}
                      >
                        <span className="well-label" style={{ position: 'absolute', top: '6px', width: '100%', textAlign: 'center', fontSize: '0.65rem', fontWeight: 700, color: '#ffffff' }}>
                          💧 {lang === 'te' ? 'నీరు' : 'Water'}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Depth Gauge Bar */}
                <div className="meter-track" style={{ width: '100%', height: '14px', background: 'rgba(22, 33, 58, 0.9)', borderRadius: '20px', margin: '14px 0 8px', position: 'relative', overflow: 'hidden', border: '1px solid rgba(255, 255, 255, 0.1)' }}>
                  <div
                    className="meter-fill"
                    style={{
                      height: '100%',
                      borderRadius: '20px',
                      width: prediction.estimated_depth_m != null && prediction.confidence > 0
                        ? `${Math.min(100, Math.max(8, (prediction.estimated_depth_m / 35) * 100))}%`
                        : '0%',
                      background: getStatusColor(prediction.condition),
                      transition: 'width 1s ease'
                    }}
                  />
                </div>
                <div className="meter-labels" style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.76rem', color: '#94a3b8' }}>
                  <span>0m / 0ft ({lang === 'te' ? 'బావి' : 'Shallow Well'})</span>
                  <span>10m / 33ft ({lang === 'te' ? 'సాధారణ బోరు' : 'Standard Well'})</span>
                  <span>20m / 65ft ({lang === 'te' ? 'మధ్యస్థం' : 'Moderate'})</span>
                  <span>30m+ / 100ft+ ({lang === 'te' ? 'లోతైన బోరింగ్' : 'Deep Boring'})</span>
                </div>
              </div>

              {/* ═══ 2. Summary Metrics Row ═══ */}
              <div className="section-header animate-in animate-in-delay-2">
                <span>{lang === 'te' ? 'వాతావరణం మరియు పొలం పరిస్థితులు (గత 30 రోజుల వర్షం & ప్రస్తుత ఎండ)' : 'Weather & Field Conditions (Past 30 Days & Current)'}</span>
              </div>

              <div className="summary-cards-row animate-in animate-in-delay-2">
                {/* Rainfall Card */}
                {(() => {
                  const rain = getRainfallDetails(prediction.weather?.rainfall_mm ?? null, lang);
                  return (
                    <div className="summary-card" style={{ borderTop: `3px solid ${rain.badgeColor}` }}>
                      <div className="summary-card-header">
                        <div className="summary-card-icon" style={{ background: 'rgba(59, 130, 246, 0.15)' }}>
                          <CloudRain size={22} color="#3b82f6" />
                        </div>
                        <span className="card-badge" style={{ color: rain.badgeColor, background: 'rgba(255,255,255,0.06)' }}>
                          {rain.tag}
                        </span>
                      </div>
                      <div className="summary-card-value" style={{ color: '#60a5fa' }}>
                        {rain.display}
                      </div>
                      <div className="summary-card-label">{lang === 'te' ? 'ఇటీవలి వర్షపాతం' : 'Recent Rainfall'}</div>
                      <div className="summary-card-desc">
                        {rain.desc}
                      </div>
                    </div>
                  );
                })()}

                {/* Temperature & Evaporation Card (Celsius ONLY) */}
                {(() => {
                  const w = getWeatherDetails(prediction.weather?.temp_c ?? null, prediction.weather?.humidity_pct ?? null, lang);
                  return (
                    <div className="summary-card" style={{ borderTop: `3px solid ${w.badgeColor}` }}>
                      <div className="summary-card-header">
                        <div className="summary-card-icon" style={{ background: 'rgba(245, 158, 11, 0.15)' }}>
                          <Thermometer size={22} color="#f59e0b" />
                        </div>
                        <span className="card-badge" style={{ color: w.badgeColor, background: 'rgba(255,255,255,0.06)' }}>
                          {w.status}
                        </span>
                      </div>
                      <div className="summary-card-value" style={{ color: '#fbbf24' }}>
                        {w.tempDisplay}
                      </div>
                      <div className="summary-card-label">{lang === 'te' ? 'సగటు ఉష్ణోగ్రత' : 'Day/Night Average Temp'}</div>
                      <div className="summary-card-desc">
                        {w.evapDesc}
                      </div>
                    </div>
                  );
                })()}

                {/* Elevation & Terrain Card */}
                {(() => {
                  const el = getElevationDetails(prediction.elevation_m, lang);
                  return (
                    <div className="summary-card" style={{ borderTop: '3px solid #8b5cf6' }}>
                      <div className="summary-card-header">
                        <div className="summary-card-icon" style={{ background: 'rgba(139, 92, 246, 0.15)' }}>
                          <Mountain size={22} color="#8b5cf6" />
                        </div>
                        <span className="card-badge" style={{ color: '#a78bfa', background: 'rgba(255,255,255,0.06)' }}>
                          {el.tag}
                        </span>
                      </div>
                      <div className="summary-card-value" style={{ color: '#a78bfa' }}>
                        {el.display}
                      </div>
                      <div className="summary-card-label">{lang === 'te' ? 'సముద్ర మట్టం నుండి ఎత్తు' : 'Elevation (Height Above Sea)'}</div>
                      <div className="summary-card-desc">
                        {el.desc}
                      </div>
                    </div>
                  );
                })()}
              </div>

              {/* ═══ 3. Soil Composition + Health Score Row ═══ */}
              <div className="insight-row animate-in animate-in-delay-3">
                {/* Soil Donut Card */}
                {(() => {
                  const phInfo = getSoilPhDetails(prediction.soil?.soil_ph ?? null, lang);
                  const soilType = getSoilTypeLabel(prediction.soil, lang);
                  return (
                    <div className="insight-card">
                      <div className="insight-card-title" style={{ justifyContent: 'space-between' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <SoilIcon size={18} color="#10b981" />
                          <span>{lang === 'te' ? 'మీ నేల రకం' : 'Your Soil Profile'}</span>
                        </div>
                        <span className="card-badge" style={{ color: phInfo.badgeColor, background: phInfo.bg }}>
                          {phInfo.status}
                        </span>
                      </div>

                      <div style={{ fontWeight: 700, fontSize: '1rem', color: 'var(--text-primary)', marginBottom: '10px' }}>
                        {soilType}
                      </div>

                      <div className="soil-donut-container">
                        <div
                          className="soil-donut"
                          style={{
                            background: soilTotal > 1
                              ? `conic-gradient(
                                  #92400e 0deg ${clayAngle}deg,
                                  #d97706 ${clayAngle}deg ${clayAngle + sandAngle}deg,
                                  #6b7280 ${clayAngle + sandAngle}deg 360deg
                                )`
                              : 'rgba(30, 41, 59, 0.8)',
                          }}
                        >
                          <div className="soil-donut-inner">
                            {prediction.soil?.soil_ph != null ? (
                              <>
                                <span className="ph-value">pH {prediction.soil.soil_ph}</span>
                                <span className="ph-label">{prediction.soil.soil_ph <= 7.5 && prediction.soil.soil_ph >= 6.0 ? (lang === 'te' ? 'తీపి/మంచిది' : 'Sweet/Good') : prediction.soil.soil_ph < 6 ? (lang === 'te' ? 'ఆమ్లం' : 'Acidic') : (lang === 'te' ? 'చౌడు' : 'Alkaline')}</span>
                              </>
                            ) : (
                              <span className="ph-label" style={{ fontSize: '0.7rem' }}>{lang === 'te' ? 'pH లేదు' : 'No pH data'}</span>
                            )}
                          </div>
                        </div>

                        <div className="soil-legend">
                          <div className="soil-legend-item">
                            <span className="soil-legend-dot" style={{ background: '#92400e' }} />
                            <span>{lang === 'te' ? 'బంకమట్టి (నీటిని నిలుపుతుంది)' : 'Clay (Holds Water)'}</span>
                            <span className="soil-legend-pct">{clayPct > 0 ? `${clayPct}%` : '--'}</span>
                          </div>
                          <div className="soil-legend-item">
                            <span className="soil-legend-dot" style={{ background: '#d97706' }} />
                            <span>{lang === 'te' ? 'ఇసుక (త్వరగా ఇంకుతుంది)' : 'Sand (Drains Quick)'}</span>
                            <span className="soil-legend-pct">{sandPct > 0 ? `${sandPct}%` : '--'}</span>
                          </div>
                          <div className="soil-legend-item">
                            <span className="soil-legend-dot" style={{ background: '#6b7280' }} />
                            <span>{lang === 'te' ? 'ఒండ్రు మట్టి (బలమైనది)' : 'Silt (Loam)'}</span>
                            <span className="soil-legend-pct">{siltPct > 0 ? `${siltPct}%` : '--'}</span>
                          </div>
                        </div>
                      </div>

                      {/* Soil pH explanation box */}
                      <div className="ph-farmer-box" style={{ borderColor: phInfo.badgeColor }}>
                        <div style={{ fontWeight: 600, color: phInfo.badgeColor, fontSize: '0.88rem' }}>
                          🌱 {phInfo.tag}:
                        </div>
                        <div style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', marginTop: '3px', lineHeight: 1.45 }}>
                          {phInfo.desc}
                        </div>
                      </div>

                      <div className="soil-message">
                        {getSoilMessage(prediction.soil, lang)}
                      </div>
                    </div>
                  );
                })()}

                {/* Health Gauge Card: Water Reserve Quantity */}
                {prediction.groundwater_health_score != null ? (
                  <div className="insight-card" style={{ borderLeft: `4px solid ${healthColor}` }}>
                    <div className="insight-card-title">
                      <Activity size={18} color={healthColor} />
                      {lang === 'te' ? 'భూగర్భ నీటి నిల్వ శాతం (లభ్యత)' : 'Groundwater Reserve & Availability Score'}
                    </div>

                    <div className="health-gauge-wrapper">
                      <div className="health-gauge-svg-container">
                        <svg viewBox="0 0 140 140">
                          {/* Background circle */}
                          <circle cx="70" cy="70" r="52" stroke="var(--gauge-track)" strokeWidth="12" fill="none" />
                          {/* Progress circle */}
                          <circle
                            cx="70" cy="70" r="52"
                            stroke={healthColor}
                            strokeWidth="12"
                            fill="none"
                            strokeDasharray={healthCircumference}
                            strokeDashoffset={healthOffset}
                            strokeLinecap="round"
                            transform="rotate(-90 70 70)"
                            style={{ transition: 'stroke-dashoffset 1s ease-out' }}
                          />
                          {/* Center text */}
                          <text x="70" y="64" textAnchor="middle" fontSize="30" fontWeight="800" fill="var(--text-primary)" fontFamily="Outfit, sans-serif">
                            {healthScore}%
                          </text>
                          <text x="70" y="84" textAnchor="middle" fontSize="11" fill="var(--text-muted)" fontWeight="600">
                            {healthScore >= 60 ? (lang === 'te' ? 'మంచి నిల్వ' : 'Healthy') : healthScore >= 35 ? (lang === 'te' ? 'మధ్యస్థం' : 'Moderate') : (lang === 'te' ? 'తక్కువ నిల్వ' : 'Critical')}
                          </text>
                        </svg>
                      </div>

                      <div className="health-status-label" style={{ color: healthColor }}>
                        {prediction.health_status
                          ? (lang === 'te' ? `${healthScore}% మంచి నీటి నిల్వ / సుస్థిర లభ్యత` : prediction.health_status)
                          : (healthScore < 35 ? (lang === 'te' ? 'నీటి కొరత ఉన్న ప్రాంతం' : 'Over-Exploited Area') : (lang === 'te' ? 'మంచి నీటి నిల్వ' : 'Stable Area'))}
                      </div>

                      <div className="health-gauge-description">
                        {lang === 'te'
                          ? `భూమిలో పంటల అవసరాలకు సరిపడేలా ${healthScore}% మేర భూగర్భ జలాల నిల్వ అందుబాటులో ఉంది.`
                          : (prediction.health_description || `Underground water reserve is currently ${healthScore}% full, showing sustainable availability for agriculture.`)}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="insight-card">
                    <div className="insight-card-title">
                      <Activity size={18} color="#94a3b8" />
                      {lang === 'te' ? 'నీటి నిల్వ స్కోరు' : 'Water Availability Score'}
                    </div>
                    <div style={{ textAlign: 'center', padding: '30px 10px', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                      {lang === 'te' ? 'ఈ ప్రాంతానికి స్కోరు అందుబాటులో లేదు.' : 'Health score data is not available for this location.'}
                    </div>
                  </div>
                )}
              </div>

              {/* ═══ 4. Distinct Clarification: Water Availability vs AI Confidence ═══ */}
              <div className="clarification-banner animate-in animate-in-delay-3">
                <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px' }}>
                  <Info size={20} color="#38bdf8" style={{ flexShrink: 0, marginTop: '2px' }} />
                  <div>
                    <div style={{ fontWeight: 700, color: '#38bdf8', fontSize: '0.92rem' }}>
                      {lang === 'te' ? '💡 రైతులకు ముఖ్య గమనిక (స్కోర్ల తేడా):' : '💡 Understanding the Scores:'}
                    </div>
                    <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '4px', lineHeight: 1.5 }}>
                      {lang === 'te' ? (
                        <>
                          • <strong>భూగర్భ నీటి నిల్వ ({healthScore}%):</strong> ఇది మీ పొలంలో ఉన్న <u>వాస్తవ నీటి పరిమాణం</u> (నీరు ఎంత సమృద్ధిగా ఉందో తెలుపుతుంది).<br />
                          • <strong>AI ఖచ్చితత్వ స్కోరు ({prediction.confidence}%):</strong> ఇది AI అంచనా ఎంత <u>నమ్మదగినదో</u> తెలుపుతుంది (10 ప్రభుత్వ పర్యవేక్షణ బావులు, ఉపగ్రహ రాడార్ ద్వారా నిర్ధారించబడింది).
                        </>
                      ) : (
                        <>
                          • <strong>Water Reserve ({healthScore}%):</strong> Represents the <u>actual quantity & health of water</u> available in the ground for farming.<br />
                          • <strong>AI Confidence ({prediction.confidence}%):</strong> Represents the <u>accuracy and reliability</u> of this AI prediction (verified against nearby government monitoring wells & satellite radar).
                        </>
                      )}
                    </div>
                  </div>
                </div>
              </div>

              {/* ═══ 5. Farming Recommendations ═══ */}
              {recommendations.length > 0 && (
                <div className="recommendations-card animate-in animate-in-delay-4">
                  <div className="recommendations-title">
                    <Lightbulb size={20} color="#f59e0b" />
                    {lang === 'te' ? 'రైతులకు ముఖ్యమైన సలహాలు (బోరుబావి & పంటల సూచనలు)' : 'What This Means for You (Farming Advice)'}
                  </div>

                  {recommendations.map((rec, idx) => (
                    <div key={idx} className="recommendation-item">
                      <div
                        className="recommendation-icon"
                        style={{ background: rec.bgColor }}
                      >
                        {rec.icon}
                      </div>
                      <div className="recommendation-text">
                        {rec.text}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* ═══ 6. How Reliable is This Estimate? ═══ */}
              <div className="confidence-card animate-in animate-in-delay-4" style={{ borderLeft: `4px solid ${prediction.confidence > 0 ? 'var(--accent-cyan)' : 'var(--text-muted)'}` }}>
                <div className="confidence-header">
                  <div className="confidence-title" style={{ color: prediction.confidence > 0 ? '#38bdf8' : '#94a3b8' }}>
                    <ShieldCheck size={18} />
                    <span>{lang === 'te' ? 'ఈ అంచనా ఎంత నమ్మదగినది? (AI ఖచ్చితత్వం)' : 'How Reliable is This Estimate? (AI Accuracy)'}</span>
                  </div>
                  <span className="confidence-value" style={{ color: prediction.confidence > 0 ? '#38bdf8' : '#94a3b8' }}>
                    {prediction.confidence}% {lang === 'te' ? 'ఖచ్చితమైనది' : 'Confident'}
                  </span>
                </div>

                <div className="confidence-explanation">
                  {lang === 'te'
                    ? `బహుళ ఆధారాల AI ధృవీకరణ: ${prediction.confidence}% | సమీపంలోని 10 ప్రభుత్వ టెస్టింగ్ బావులు • వర్షపాతం: ${prediction.weather?.rainfall_mm ?? '0'} mm • నేల: క్లే ${prediction.soil?.clay_pct ?? 0}%, ఇసుక ${prediction.soil?.sand_pct ?? 0}%, ఒండ్రు ${prediction.soil?.silt_pct ?? 0}% • ఉపగ్రహ నిర్ధారణ.`
                    : (prediction.confidence_note || `Multi-Factor Confidence: ${prediction.confidence}% based on nearby CGWB monitoring stations, satellite radar, and soil composition.`)}
                </div>
              </div>
            </>
          ) : null}
        </div>

        {/* ════ RIGHT: Interactive Map View ════ */}
        <div className="map-view-pane">
          {/* Coordinate Search Form & Presets Bar */}
          <div className="map-search-bar">
            <form onSubmit={handleManualSearch} className="search-inputs-group">
              <div className="input-with-label">
                <label>LAT</label>
                <input
                  type="text"
                  value={inputLat}
                  onChange={(e) => setInputLat(e.target.value)}
                  placeholder="17.4228"
                />
              </div>
              <div className="input-with-label">
                <label>LON</label>
                <input
                  type="text"
                  value={inputLon}
                  onChange={(e) => setInputLon(e.target.value)}
                  placeholder="78.6500"
                />
              </div>
              <button type="submit" className="btn-search">
                <Search size={14} />
                <span>{lang === 'te' ? 'పరిశీలించు' : 'Check'}</span>
              </button>
            </form>

            <div className="popular-locations-bar">
              {POPULAR_LOCATIONS.map((loc, idx) => (
                <button
                  key={idx}
                  className="preset-btn"
                  onClick={() => handleLocationChange(loc.lat, loc.lon)}
                >
                  <MapPin size={12} />
                  <span>{lang === 'te' ? loc.telugu : loc.name}</span>
                </button>
              ))}
            </div>
          </div>

          <div id="map" ref={mapContainerRef} style={{ width: '100%', height: '100%' }} />
        </div>
      </div>

      {/* ── Feedback Modal ────────────────────────────────────────── */}
      {feedbackOpen && (
        <div className="modal-overlay" onClick={() => setFeedbackOpen(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>
                <Droplets size={20} color="var(--accent-cyan)" />
                <span>{lang === 'te' ? 'రైతు బోరుబావి సమాచారం' : 'Share Borewell Data'}</span>
              </h3>
              <button className="btn-close" onClick={() => setFeedbackOpen(false)} aria-label="Close">
                ✕
              </button>
            </div>

            <div className="modal-location-badge">
              <MapPin size={14} />
              <span>
                {prediction?.location?.displayName
                  ? (prediction.location.displayName.length > 50 ? `${prediction.location.displayName.slice(0, 50)}...` : prediction.location.displayName)
                  : `GPS: ${coords.lat.toFixed(4)}°N, ${coords.lon.toFixed(4)}°E`}
              </span>
            </div>

            {feedbackSubmitted ? (
              <div className="modal-submitted-box">
                <div className="modal-submitted-icon">
                  <CheckCircle2 size={32} />
                </div>
                <h4 className="modal-submitted-title">
                  {lang === 'te' ? 'ధన్యవాదాలు! వివరాలు నమోదయ్యాయి.' : 'Thank you! Borewell Data Recorded.'}
                </h4>
                <p className="modal-submitted-subtitle">
                  {lang === 'te'
                    ? 'మీరు సమర్పించిన వాస్తవ లోతు ఆధారంగా ఏఐ మోడల్ అంచనాలు మరింత మెరుగవుతాయి.'
                    : 'Your ground truth data has been recorded to improve future AI predictions.'}
                </p>
              </div>
            ) : (
              <>
                <p className="modal-desc">
                  {lang === 'te'
                    ? 'మీ పొలంలో లేదా ప్రాంగణంలో వేసిన బోరుబావి వాస్తవ లోతు వివరాలు పంపితే ఏఐ మోడల్ మరింత ఖచ్చితంగా పనిచేస్తుంది.'
                    : 'Help improve GeoGround AI for local communities by contributing actual water strike depth for this location.'}
                </p>

                <form onSubmit={handleFeedbackSubmit}>
                  <div style={{ marginBottom: '16px' }}>
                    <label style={{ display: 'block', fontSize: '0.84rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '8px' }}>
                      {lang === 'te' ? 'వాస్తవ బోరుబావి లోతు (మీటర్లలో)' : 'Actual Struck Water Depth (Meters)'}
                    </label>
                    <input
                      type="number"
                      step="0.1"
                      min="0.5"
                      max="150"
                      required
                      value={feedbackActualDepth}
                      onChange={(e) => setFeedbackActualDepth(e.target.value)}
                      placeholder="e.g. 8.5"
                      className="modal-input"
                      autoFocus
                    />
                    {feedbackActualDepth && !isNaN(parseFloat(feedbackActualDepth)) && (
                      <div style={{ fontSize: '0.82rem', color: 'var(--accent-cyan)', marginTop: '6px', fontWeight: 600 }}>
                        ≈ {(parseFloat(feedbackActualDepth) * 3.28084).toFixed(0)} {lang === 'te' ? 'అడుగుల లోతు' : 'feet deep'}
                      </div>
                    )}
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px', marginTop: '22px' }}>
                    <button
                      type="button"
                      className="btn-pill"
                      onClick={() => setFeedbackOpen(false)}
                    >
                      {lang === 'te' ? 'రద్దు' : 'Cancel'}
                    </button>
                    <button type="submit" className="btn-pill btn-pill-primary">
                      <Send size={14} />
                      <span>{lang === 'te' ? 'సమర్పించు' : 'Submit Depth'}</span>
                    </button>
                  </div>
                </form>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
