import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import re
import string
import unicodedata
from collections import Counter
import csv

data_path = Path(__file__).parent
reviews = pd.read_csv(data_path / "tCF Review Data - reviews_last_8y.csv")
reviews["text"] = reviews["text"].astype(str)

# Identify empty reviews
reviews["is_empty"] = reviews["text"].apply(lambda x: len(x.strip()) == 0)
print("Empty reviews:", sum(reviews["is_empty"]))
print("Non-empty reviews:", sum(~reviews["is_empty"]))

print(reviews[reviews["is_empty"]]["text"].head())

# Use only reviews with non-empty text
reviews = reviews[~reviews["is_empty"]]

# remove any characters that aren't standard English letters, digits, punctuation or whitespace
def clean_text_keep_ascii_punct(s: str) -> str:
    if s is None:
        return s
    # decompose unicode (remove accents), then drop non-ASCII
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    # keep letters, digits, standard punctuation, and whitespace; remove everything else
    allowed_punct = re.escape(string.punctuation)
    s = re.sub(rf"[^A-Za-z0-9{allowed_punct}\s]+", "", s)
    # collapse multiple whitespace to single space and strip
    s = re.sub(r"\s+", " ", s).strip()
    return s

reviews["text"] = reviews["text"].apply(clean_text_keep_ascii_punct)

# 1) Number of reviews per semester (combine semester_season + semester_year)
# Prepare semester labels and sort order
season_order_map = {"Winter": 1, "Spring": 2, "Summer": 3, "Fall": 4}
# fall back to alphabetical order code if season not in map
reviews["season_order"] = reviews["semester_season"].map(season_order_map).fillna(
    reviews["semester_season"].astype(str).str.lower().apply(lambda s: sum(ord(c) for c in s) % 10 + 1)
)
# Ensure year is int where possible
reviews["semester_year"] = reviews["semester_year"].astype(str)
reviews["semester_year_int"] = pd.to_numeric(reviews["semester_year"], errors="coerce").fillna(0).astype(int)
reviews["semester_label"] = reviews["semester_season"].astype(str) + " " + reviews["semester_year"].astype(str)
reviews["semester_sort_key"] = reviews["semester_year_int"] * 10 + reviews["season_order"]
sem_counts = (
    reviews.groupby(["semester_label", "semester_sort_key"])
    .size()
    .reset_index(name="count")
    .sort_values("semester_sort_key")
)
semester_order = sem_counts["semester_label"].tolist()
plt.figure(figsize=(10, 5))
sns.barplot(x="semester_label", y="count", data=sem_counts, order=semester_order)
plt.title("Number of Reviews per Semester")
plt.xlabel("Semester")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("reviews_per_semester.png")

# 2) Number of reviews per department (first token of course_code)
# Extract department (first word/token of course_code)
reviews["department"] = reviews["course_code"].astype(str).str.split().str[0].fillna("UNKNOWN")
dept_counts = reviews["department"].value_counts().reset_index()
dept_counts.columns = ["department", "count"]
# Order departments by descending count for the bar plot
dept_order = dept_counts.sort_values("count", ascending=False)["department"].tolist()
plt.figure(figsize=(15, 6))
sns.barplot(x="department", y="count", data=dept_counts, order=dept_order)
plt.title("Number of Reviews per Department")
plt.xlabel("Department")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=90, ha="right")
plt.tight_layout()
plt.savefig("reviews_per_department.png")

# compute top 100 common words from reviews["text"]
corpus = " ".join(reviews["text"].dropna().astype(str).str.lower())
# tokenize: keep alphabetic tokens and internal apostrophes, require length > 1 after stripping quotes
tokens = re.findall(r"[a-z']+", corpus)
tokens = [t.strip("'") for t in tokens if len(t.strip("'")) > 1]

# stopwords: prefer wordcloud STOPWORDS if available, otherwise fallback to a small set
try:
    from wordcloud import STOPWORDS as WC_STOPWORDS
    stopwords = set(w.lower() for w in WC_STOPWORDS)
except Exception:
    stopwords = {
        "the", "and", "to", "a", "of", "in", "for", "is", "it", "on", "this",
        "that", "i", "was", "with", "but", "as", "are", "they", "be", "course",
        "class", "students", "professor", "teacher", "professors", "student"
    }

tokens = [t for t in tokens if t not in stopwords]

counter = Counter(tokens)
top100 = counter.most_common(100)

# print to console
print("Top 20 words:")
for w, c in top100:
    print(f"{w}: {c}")

# make a horizontal bar plot of the most common words (from top100)
top_df = pd.DataFrame(top100, columns=["word", "count"])
# determine figure height dynamically
height = max(6, len(top_df) * 0.25)
plt.figure(figsize=(10, height))
sns.barplot(x="count", y="word", data=top_df.sort_values("count", ascending=False))
plt.title("Top Words in Reviews")
plt.xlabel("Count")
plt.ylabel("Word")
plt.tight_layout()
plt.savefig("top_words_bar.png")
plt.show()