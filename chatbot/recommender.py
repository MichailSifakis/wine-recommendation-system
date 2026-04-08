"""
Content-Based Wine Recommender
================================
Builds TF-IDF vectors from wine metadata and finds wines
most similar to a user preference query via cosine similarity.

Features used:
  Type | Grapes | Harmonize (food pairings) | Body | Acidity | Country | RegionName | Elaborate
"""

import re
import ast
import numpy as np
import pandas as pd
import gdown
import tempfile
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dataframe(path_or_url: str) -> pd.DataFrame:
    """Helper to load CSV from local path or Google Drive directly."""
    if "drive.google.com" in path_or_url:
        print("[Recommender] Detected Google Drive URL. Downloading temporary dataset...")
        file_id = path_or_url.split("id=")[-1]
        tmp = tempfile.mktemp(suffix=".csv")
        gdown.download(id=file_id, output=tmp, quiet=False)
        df = pd.read_csv(tmp, low_memory=False)
        os.remove(tmp)
        return df
    return pd.read_csv(path_or_url, low_memory=False)

def _safe_parse_list(value: str) -> list[str]:
    """Parse a Python-list-encoded string like "['Beef', 'Pizza']" safely."""
    if not isinstance(value, str) or value.strip() == "":
        return []
    try:
        result = ast.literal_eval(value)
        if isinstance(result, list):
            return [str(x).lower().strip() for x in result]
    except Exception:
        pass
    # Fallback: strip brackets and split on comma
    cleaned = re.sub(r"[\[\]']", "", value)
    return [x.lower().strip() for x in cleaned.split(",") if x.strip()]


def _build_feature_string(row: pd.Series) -> str:
    """
    Combine all relevant metadata fields into a single text document.
    Repeated tokens act as TF-IDF weight boosters.
    E.g. Type is repeated 3x so it has more influence than Region.
    """
    parts = []

    # Wine type (high weight → repeat)
    wine_type = str(row.get("Type", "")).lower().strip()
    parts.extend([wine_type] * 3)

    # Grapes
    grapes = _safe_parse_list(str(row.get("Grapes", "")))
    parts.extend(grapes * 2)

    # Food pairings (Harmonize)
    harmonize = _safe_parse_list(str(row.get("Harmonize", "")))
    parts.extend(harmonize)

    # Body & Acidity
    body = str(row.get("Body", "")).lower().strip()
    acidity = str(row.get("Acidity", "")).lower().strip()
    parts.extend([body] * 2)
    parts.append(acidity)

    # Country & Region
    country = str(row.get("Country", "")).lower().strip()
    region = str(row.get("RegionName", "")).lower().strip()
    parts.extend([country] * 2)
    parts.append(region)

    # Elaborate (varietal style: "Varietal/100%", "Blend", etc.)
    elaborate = str(row.get("Elaborate", "")).lower().replace("/", " ").strip()
    parts.append(elaborate)

    return " ".join(t for t in parts if t and t != "nan")


# ---------------------------------------------------------------------------
# Main Recommender Class
# ---------------------------------------------------------------------------

class ContentBasedRecommender:
    """
    Loads the XWines wine CSV, builds a TF-IDF matrix,
    and exposes a recommend() method.
    """

    def __init__(self, csv_path: str):
        print(f"[Recommender] Loading wine data from {csv_path} …")
        self.df = load_dataframe(csv_path)
        self._clean()
        self._build_index()
        print(f"[Recommender] Ready — {len(self.df):,} wines indexed.")

    # ------------------------------------------------------------------
    # Internal setup
    # ------------------------------------------------------------------

    def _clean(self):
        """Basic cleaning: fill missing values, normalise column names."""
        self.df = self.df.fillna("")
        # Ensure numeric ABV
        self.df["ABV"] = pd.to_numeric(self.df["ABV"], errors="coerce").fillna(0.0)

    def _build_index(self):
        """Build the TF-IDF matrix over all wines."""
        print("[Recommender] Building TF-IDF index …")
        self.df["_feature_str"] = self.df.apply(_build_feature_string, axis=1)
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),   # unigrams + bigrams (captures "full bodied", "pinot noir")
            min_df=2,             # ignore very rare tokens
            max_df=0.95,          # ignore extremely common tokens
            sublinear_tf=True,    # log-normalise term frequencies
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["_feature_str"])
        print(f"[Recommender] TF-IDF matrix shape: {self.tfidf_matrix.shape}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend(
        self,
        preferences: dict,
        top_n: int = 8,
    ) -> list[dict]:
        """
        Given a preference dict, return top-N similar wines.

        preferences keys (all optional):
          wine_type   : str   e.g. "Red", "White", "Sparkling", "Rosé"
          grapes      : list  e.g. ["Cabernet Sauvignon", "Merlot"]
          food        : list  e.g. ["Beef", "Pasta"]
          body        : str   e.g. "Full-bodied", "Medium-bodied", "Light-bodied"
          acidity     : str   e.g. "High", "Medium", "Low"
          country     : str   e.g. "France", "Italy"
          region      : str   e.g. "Bordeaux"
          abv_min     : float e.g. 12.0
          abv_max     : float e.g. 15.0
          elaborate   : str   e.g. "Blend", "Varietal"
          exclude_ids : list  wine IDs to exclude (already recommended)
        """
        # 1. Build a query string mirroring _build_feature_string logic
        query_parts = []

        wine_type = preferences.get("wine_type", "")
        if wine_type:
            query_parts.extend([wine_type.lower()] * 3)

        for grape in preferences.get("grapes", []):
            query_parts.extend([grape.lower()] * 2)

        for food in preferences.get("food", []):
            query_parts.append(food.lower())

        body = preferences.get("body", "")
        if body:
            query_parts.extend([body.lower()] * 2)

        acidity = preferences.get("acidity", "")
        if acidity:
            query_parts.append(acidity.lower())

        country = preferences.get("country", "")
        if country:
            query_parts.extend([country.lower()] * 2)

        region = preferences.get("region", "")
        if region:
            query_parts.append(region.lower())

        elaborate = preferences.get("elaborate", "")
        if elaborate:
            query_parts.append(elaborate.lower())

        if not query_parts:
            # No preferences — return popular sample
            return self.df.sample(top_n)[self._wine_cols()].to_dict(orient="records")

        query_str = " ".join(query_parts)

        # 2. Transform query into TF-IDF space
        query_vec = self.vectorizer.transform([query_str])

        # 3. Cosine similarity against all wines
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # 4. Hard filters on top of similarity ranking
        mask = self._build_mask(preferences)

        # 5. Apply mask: zero out filtered wines
        filtered_sim = similarities * mask

        # 6. Exclude already-recommended wines
        exclude_ids = set(preferences.get("exclude_ids", []))
        if exclude_ids:
            for idx, row in self.df.iterrows():
                if row["WineID"] in exclude_ids:
                    filtered_sim[idx] = 0.0

        # 7. Get top-N indices
        top_indices = np.argsort(filtered_sim)[::-1][:top_n]

        # 8. Build result dicts
        results = []
        for i in top_indices:
            if filtered_sim[i] <= 0:
                continue
            row = self.df.iloc[i]
            wine_dict = {col: row[col] for col in self._wine_cols()}
            wine_dict["similarity_score"] = round(float(filtered_sim[i]), 4)
            wine_dict["grapes_parsed"] = _safe_parse_list(str(row.get("Grapes", "")))
            wine_dict["food_parsed"] = _safe_parse_list(str(row.get("Harmonize", "")))
            results.append(wine_dict)

        return results

    def get_wine_by_id(self, wine_id: int) -> Optional[dict]:
        """Return full metadata for a single wine by ID."""
        match = self.df[self.df["WineID"] == wine_id]
        if match.empty:
            return None
        row = match.iloc[0]
        return {col: row[col] for col in self._wine_cols()}

    def get_available_options(self) -> dict:
        """Return distinct values for filter fields (for UI / Gemini context)."""
        return {
            "types": sorted(self.df["Type"].dropna().unique().tolist()),
            "countries": sorted(self.df["Country"].dropna().unique().tolist()),
            "bodies": sorted(self.df["Body"].dropna().unique().tolist()),
            "acidities": sorted(self.df["Acidity"].dropna().unique().tolist()),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_mask(self, preferences: dict) -> np.ndarray:
        """Return a boolean array (as float) masking wines that fail hard filters."""
        mask = np.ones(len(self.df), dtype=float)

        wine_type = preferences.get("wine_type", "")
        if wine_type:
            mask *= (self.df["Type"].str.lower() == wine_type.lower()).astype(float).values

        country = preferences.get("country", "")
        if country:
            mask *= (self.df["Country"].str.lower() == country.lower()).astype(float).values

        abv_min = preferences.get("abv_min")
        if abv_min is not None:
            mask *= (self.df["ABV"] >= float(abv_min)).astype(float).values

        abv_max = preferences.get("abv_max")
        if abv_max is not None:
            mask *= (self.df["ABV"] <= float(abv_max)).astype(float).values

        return mask

    def _wine_cols(self) -> list[str]:
        """Columns to include in result dicts."""
        return [
            "WineID", "WineName", "Type", "Elaborate",
            "Grapes", "Harmonize", "ABV", "Body", "Acidity",
            "Country", "RegionName", "WineryName", "Website",
        ]
