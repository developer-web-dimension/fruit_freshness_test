#include <Wire.h>
#include <SparkFun_AS7265X.h>

AS7265X sensor;

// Capture
const int   NUM_FRAMES           = 3;          // average N frames
const float SAT_CUTOFF           = 3000.0f;    // simple saturation gate
const float MIN_TOTAL_SIGNAL     = 60.0f;      // reject very weak scenes
const uint8_t INTEG_CYCLES       = 60;         // exposure (cycles)
const uint8_t GAIN_SETTING       = AS7265X_GAIN_16X;
const uint8_t LED_CURRENT        = AS7265X_LED_CURRENT_LIMIT_25MA;

// ----- Apple-specific thresholds derived from your samples -----
// G/R (higher is fresher)
const float GR_FRESH_MIN   = 0.37;  // fresh ~0.396–0.400
const float GR_STALE_MAX   = 0.33;  // stale ~0.29–0.31

// ARI' (LOWER is fresher in your data)
const float ARI_FRESH_MAX  = 0.002; // <= 0.002 → fully fresh
const float ARI_STALE_MIN  = 0.008; // >= 0.008 → fully stale

// NDWI' (LOWER is fresher in your data)
const float NDWI_FRESH_MAX = 0.49;
const float NDWI_STALE_MIN = 0.55;

// T/S (730/680) (LOWER is fresher in your data)
const float TOS_FRESH_MAX  = 0.455;
const float TOS_STALE_MIN  = 0.470;

// U/S (760/680) (LOWER is fresher in your data)
const float UOS_FRESH_MAX  = 0.255;
const float UOS_STALE_MIN  = 0.275;

// ---- Scoring weights (sum = 100) ----
const int W_GR   = 10;
const int W_ARI  = 40;
const int W_NDWI = 10;
const int W_TOS  = 5;
const int W_UOS  = 5;
// NDVI currently unreliable in your setup -> weight 0

// ====================== HELPERS ======================
static inline float safeDiv(float a, float b) { return (b != 0.0f) ? (a / b) : 0.0f; }

// Map an index into 0..20 points with tunable “fresh vs stale” bounds
// If higher_is_better = true:
//    val >= fresh_min -> 20; val <= stale_max -> 0; linear in-between.
// If higher_is_better = false (lower is better, e.g., ARI, NDWI, T/S, U/S in your data):
//    val <= fresh_max -> 20; val >= stale_min -> 0; linear in-between.
int scoreIndex(float val, float fresh_min_or_max, float stale_max_or_min, bool higher_is_better) {
  if (higher_is_better) {
    float fresh_min = fresh_min_or_max;
    float stale_max = stale_max_or_min;
    if (val >= fresh_min) return 20;
    if (val <= stale_max) return 0;
    return (int)(20.0f * (val - stale_max) / (fresh_min - stale_max));
  } else {
    float fresh_max = fresh_min_or_max;
    float stale_min = stale_max_or_min;
    if (val <= fresh_max) return 20;
    if (val >= stale_min) return 0;
    return (int)(20.0f * (stale_min - val) / (stale_min - fresh_max));
  }
}

static inline int scalePts(int pts0to20, int weight) {
  // scale 0..20 points into 0..weight
  return (pts0to20 * weight) / 20;
}

// ====================== ARDUINO ======================
void setup() {
  Serial.begin(115200);
  Wire.begin();

  if (!sensor.begin()) {
    Serial.println(F("Sensor not found!"));
    while (1);
  }

  sensor.setIntegrationCycles(INTEG_CYCLES);
  sensor.setGain(GAIN_SETTING);

  // Consistent illumination
  sensor.enableBulb(AS7265x_LED_WHITE);
  sensor.setBulbCurrent(AS7265x_LED_WHITE, LED_CURRENT);

  Serial.println(F("Fruit Freshness Analyzer – Index-based (Apple-tuned)"));
  Serial.println(F("Keep 3–5 cm distance; use a dark shroud if possible."));
  delay(1200);
}

void loop() {
  // --- Average multiple frames with a simple saturation gate ---
  float sumE=0, sumF=0, sumG=0, sumH=0, sumI=0, sumJ=0, sumS=0, sumT=0, sumU=0, sumV=0, sumW=0, sumL=0;
  int valid = 0;

  for (int i = 0; i < NUM_FRAMES; i++) {
    sensor.takeMeasurements();

    float E = sensor.getCalibratedE(); // 510 (green)
    float F = sensor.getCalibratedF(); // 535 (green)
    float G = sensor.getCalibratedG(); // 560 (green)
    float H = sensor.getCalibratedH(); // 585 (yellow)
    float I = sensor.getCalibratedI(); // 645 (red)
    float J = sensor.getCalibratedJ(); // 705 (red-edge)
    float S = sensor.getCalibratedS(); // 680 (red)
    float T = sensor.getCalibratedT(); // 730
    float U = sensor.getCalibratedU(); // 760
    float V = sensor.getCalibratedV(); // 810 (NIR)
    float W = sensor.getCalibratedW(); // 860 (NIR)
    float L = sensor.getCalibratedL(); // 940 (NIR water)

    // Saturation/overflow gate (very simple, tune if needed)
    bool ok = (E<SAT_CUTOFF && F<SAT_CUTOFF && G<SAT_CUTOFF && H<SAT_CUTOFF &&
               I<SAT_CUTOFF && J<SAT_CUTOFF && S<SAT_CUTOFF && T<SAT_CUTOFF &&
               U<SAT_CUTOFF && V<SAT_CUTOFF && W<SAT_CUTOFF && L<SAT_CUTOFF);

    if (ok) {
      sumE+=E; sumF+=F; sumG+=G; sumH+=H; sumI+=I; sumJ+=J;
      sumS+=S; sumT+=T; sumU+=U; sumV+=V; sumW+=W; sumL+=L;
      valid++;
    }
    delay(120);
  }

  if (valid < 2) {
    Serial.println(F("MEASUREMENT ERROR – Adjust position/lighting"));
    delay(1500);
    return;
  }

  // Averages
  float E = sumE/valid, F = sumF/valid, G = sumG/valid, H = sumH/valid;
  float I = sumI/valid, J = sumJ/valid, S = sumS/valid;
  float T = sumT/valid, U = sumU/valid, V = sumV/valid, W = sumW/valid, L = sumL/valid;

  // Basic validity check
  float total_sig = E+F+G+H+I+J+S+T+U+V+W+L;
  if (total_sig < MIN_TOTAL_SIGNAL) {
    Serial.println(F("NO FRUIT / Very low signal – move closer or increase illumination"));
    delay(1200);
    return;
  }

  // ---------- Compute indices ----------
  // G/R = mean(510,535,560,585) / 680
  float green_mean = (E + F + G + H) / 4.0f;    // include 585 to capture yellowing
  float GR         = safeDiv(green_mean, S);    // S=680

  // ARI’ = (1/560 - 1/705)
  float ARI        = (safeDiv(1.0f, G) - safeDiv(1.0f, J)); // G=560, J=705

  // NDVI: unreliable here -> we compute but weight=0 (you can still log it)
  float NIR        = (V + W) / 2.0f;            // mean(810, 860)
  float NDVI       = (NIR + S > 0.0f) ? ((NIR - S) / (NIR + S)) : 0.0f;

  // NDWI’ = (860 - 940) / (860 + 940)
  float NDWI       = (W + L > 0.0f) ? ((W - L) / (W + L)) : 0.0f;

  // Red-edge ratios
  float T_over_S   = safeDiv(T, S);  // 730/680
  float U_over_S   = safeDiv(U, S);  // 760/680

  // ---------- Scoring (0..100) ----------
  // Direction: GR higher=better, others lower=better (as per your measurements)
  int pts_GR   = scalePts(scoreIndex(GR,   GR_FRESH_MIN,   GR_STALE_MAX,   true),  W_GR);
  int pts_ARI  = scalePts(scoreIndex(ARI,  ARI_FRESH_MAX,  ARI_STALE_MIN,  false), W_ARI);
  int pts_NDWI = scalePts(scoreIndex(NDWI, NDWI_FRESH_MAX, NDWI_STALE_MIN, false), W_NDWI);
  int pts_TOS  = scalePts(scoreIndex(T_over_S, TOS_FRESH_MAX, TOS_STALE_MIN, false), W_TOS);
  int pts_UOS  = scalePts(scoreIndex(U_over_S, UOS_FRESH_MAX, UOS_STALE_MIN, false), W_UOS);

  // NDVI ignored (weight 0)
  int score = pts_GR + pts_ARI + pts_NDWI + pts_TOS + pts_UOS;
  score = constrain(score, 0, 100);

  // Category & probability mapping
  int probability;
  String category;
  if (score >= 60) {
    category = "FRESH";
    probability = 70 + (int)((score - 60) * 1.0f);
  } else if (score >= 45) {
    category = "AVERAGE";
    probability = 45 + (int)((score - 45) * 1.67f);
  } else {
    category = "STALE";
    probability = (int)(score * 1.0f);
  }
  if (probability > 100) probability = 100;
  if (probability < 0)   probability = 0;

  // Confidence from signal strength (simple; improve with frame variance if needed)
  int confidence = 40;
  if      (total_sig >= 2000) confidence = 95;
  else if (total_sig >= 1000) confidence = 85;
  else if (total_sig >=  400) confidence = 70;
  else if (total_sig >=  200) confidence = 55;

  // --------- Print results ----------
  Serial.print(F("FRESHNESS_SCORE:")); Serial.println(score);
  Serial.print(F("PROBABILITY:"));     Serial.print(probability); Serial.println(F("%"));
  Serial.print(F("CATEGORY:"));        Serial.println(category);
  Serial.print(F("CONFIDENCE:"));      Serial.print(confidence);  Serial.println(F("%"));

  Serial.println(F("--- Index Details ---"));
  Serial.print(F("G/R: "));    Serial.println(GR,   3);
  Serial.print(F("ARI': "));   Serial.println(ARI,  6);
  Serial.print(F("NDVI: "));   Serial.println(NDVI, 3);  // logged but not scored
  Serial.print(F("NDWI': "));  Serial.println(NDWI, 3);
  Serial.print(F("T/S: "));    Serial.println(T_over_S, 3);
  Serial.print(F("U/S: "));    Serial.println(U_over_S, 3);

  Serial.print(F("Total signal: ")); Serial.println(total_sig, 0);
  Serial.println(F("========================"));

  delay(2500);
}
