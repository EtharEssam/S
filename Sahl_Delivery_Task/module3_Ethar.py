import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
sns.set_style("whitegrid")

RAW_PATH = "Sahl_Delivery.csv"

df = pd.read_csv(RAW_PATH)
print("=" * 70)
print(" original shape :", df.shape)  # (4295, 21)

# ====================================================================
print("\n" + "=" * 70)
print("STEP 1: الاستكشاف الأولي")
print("=" * 70)

print("\n--- df.info() ---")
df.info()

print("\n--- df.isnull().sum() ---")
print(df.isnull().sum())
# distance_km 57 | actual_minutes 109 | customer_rating 842
# courier_notes 4082 | reordered_within_30d 109

print("\n--- df.describe() ---")
print(df.describe(include="all"))

print("\n--- df.duplicated().sum() ---")
print(df.duplicated().sum())  # = 91 

text_cols = df.select_dtypes(include="object").columns.tolist()
print("\n--- unique() on text columns ---")
for col in text_cols:
    print(f"\n[{col}] -> {df[col].nunique()} unique values")
    print(df[col].unique()[:30])

# ====================================================================
print("\n" + "=" * 70)
print("STEP 2: cleaning & preprocessing")
print("=" * 70)
#duplicates, nulls, data types, and text cleaning
before = df.shape[0]
df = df.drop_duplicates()
print(f"[1] duplicates removed : {before - df.shape[0]}   (91)")

null_ratio = df.isnull().mean()
cols_to_drop = null_ratio[null_ratio > 0.90].index.tolist()
print(f"[2] columns to be dropped (null > 90%): {cols_to_drop}")
df = df.drop(columns=cols_to_drop)

# "EGP 564" / "60 EGP" / "1,161 EGP" -> 564 / 60 / 1161
def clean_numeric_text(series):
    series = series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    series = series.replace("", np.nan)
    return pd.to_numeric(series, errors="coerce")

df["order_value_egp"] = clean_numeric_text(df["order_value_egp"])
print("[3] order_value_egp :")
print(df["order_value_egp"].describe())

# Minya / New Minya / Abu Qurqas
# (فروق كابيتال، مسافات زيادة، أخطاء إملائية زي Menia/Abu Qorqas，
def clean_city(value):
    s = str(value).strip().lower().replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if "new minya" in s or "gedida" in s:
        return "New Minya"
    if "abu q" in s:  # Abu Qurqas & Abu Qorqas 
        return "Abu Qurqas"
    return "Minya"

before_unique = df["city"].nunique()
df["city"] = df["city"].apply(clean_city)
print(f"[4] city: {before_unique} unique values -> {df['city'].nunique()} (Minya / New Minya / Abu Qurqas)")
print(df["city"].value_counts())
# M/D/YYYY H:MM  (زي "5/15/2025 18:43")
# D/M/YYYY H:MM  (زي "25/06/2025 22:06" - اليوم أكبر من 12)
# --------------------------------------------------------------
raw_dates = df["order_datetime"].astype(str)

pass1 = pd.to_datetime(raw_dates, format="%m/%d/%Y %H:%M", errors="coerce")
still_missing = pass1.isna()
pass2 = pd.to_datetime(raw_dates[still_missing], format="%d/%m/%Y %H:%M", errors="coerce")
pass1.loc[still_missing] = pass2

df["order_datetime"] = pass1
print(f"[5] order_datetime: عدد الفشل النهائي (NaT) = {df['order_datetime'].isna().sum()}  (المتوقع 0)")

#negative actual_minutes
# --------------------------------------------------------------
before = df.shape[0]
impossible_mask = df["actual_minutes"] < 0
print(f"[6] صفوف بـ actual_minutes سالب: {impossible_mask.sum()}  (المتوقع 39)")
df = df[~impossible_mask]
print(f"    صفوف اتحذفت فعليًا: {before - df.shape[0]}")

#non-sensical distance_km (negative or zero)
# --------------------------------------------------------------
n_missing_dist = df["distance_km"].isna().sum()
median_dist = df["distance_km"].median()
df["distance_km"] = df["distance_km"].fillna(median_dist)
print(f"[7] distance_km: اتملى {n_missing_dist} قيمة بالـ median = {median_dist}")

#customer_rating missing values
# --------------------------------------------------------------
print(f"[8] customer_rating: {df['customer_rating'].isna().sum()} صف من غير تقييم -- سايبينهم فاضيين عمدًا")
# ====================================================================
print("\n" + "=" * 70)
print("CHECKPOINT")
print("=" * 70)
print("df.shape:", df.shape)
print("df.duplicated().sum():", df.duplicated().sum())  #zero
print("df.isnull().sum():\n", df.isnull().sum())

# STEP 3 — Feature Engineering
# ====================================================================
print("\n" + "=" * 70)
print("STEP 3: Feature Engineering")
print("=" * 70)
df["delay_minutes"] = df["actual_minutes"] - df["promised_minutes"]
df["is_late"] = df["delay_minutes"] > 0
df["order_hour"] = df["order_datetime"].dt.hour
df["order_day_name"] = df["order_datetime"].dt.day_name()
df["has_rating"] = df["customer_rating"].notna()
print("تم إنشاء: delay_minutes, is_late, order_hour, order_day_name, has_rating")
# STEP 4 — Investigation
# ====================================================================
print("\n" + "=" * 70)
print("STEP 4: Investigation")
print("=" * 70)
delivered = df[df["order_status"] == "Delivered"].copy()
print(f"عدد الطلبات Delivered المستخدمة في التحليل: {delivered.shape[0]} من أصل {df.shape[0]}")
reorder_rate = delivered["reordered_within_30d"].mean() * 100
print(f"\n[س1] نسبة العملاء اللي رجعوا خلال 30 يوم: {reorder_rate:.1f}%")
plt.figure(figsize=(5, 4))
delivered["reordered_within_30d"].value_counts().sort_index().plot(
    kind="bar", color=["#d9534f", "#5cb85c"]
)
plt.title("Reorder within 30 Days (0=No, 1=Yes)")
plt.xlabel("Reordered")
plt.ylabel("Number of Orders")
plt.tight_layout()
plt.savefig("q1_reorder_overview.png")
plt.close()

#on-time rate & average delay
on_time_rate = (delivered["delay_minutes"] <= 0).mean() * 100
avg_delay = delivered["delay_minutes"].mean()
print("Q2: On-time delivery rate =", round(on_time_rate, 1), "%")
print("Q2: Average delay =", round(avg_delay, 1), "minutes")
plt.figure(figsize=(6, 4))
sns.histplot(delivered["delay_minutes"], bins=40, kde=True)
plt.axvline(0, color="red", linestyle="--", label="الميعاد الموعود")
plt.title("Delivery Delay Distribution (minutes)")
plt.xlabel("Delay = actual_minutes - promised_minutes")
plt.ylabel("Number of Orders")
plt.legend()
plt.tight_layout()
plt.savefig("q2_delay_distribution.png")
plt.close()
#reorder rate: late vs on-time
reorder_by_late = delivered.groupby("is_late")["reordered_within_30d"].mean() * 100
print("\n[س3] معدل الرجوع: متأخر مقابل في الميعاد:")
print(reorder_by_late)

plt.figure(figsize=(5, 4))
reorder_by_late.plot(kind="bar", color=["#5cb85c", "#d9534f"])
plt.title("Reorder Rate: On-time vs Late Orders")
plt.xlabel("Was the order late?")
plt.ylabel("Reorder Rate (%)")
plt.tight_layout()
plt.savefig("q3_reorder_vs_late.png")
plt.close()
#how reorder rate varies by vehicle type, cuisine, and payment method
reorder_by_vehicle = delivered.groupby("vehicle_type")["reordered_within_30d"].mean() * 100
reorder_by_cuisine = delivered.groupby("cuisine")["reordered_within_30d"].mean() * 100
reorder_by_payment = delivered.groupby("payment_method")["reordered_within_30d"].mean() * 100

print("\n[س4-أ] الرجوع حسب نوع المركبة:\n", reorder_by_vehicle)
print("\n[س4-ب] الرجوع حسب نوع الأكل:\n", reorder_by_cuisine)
print("\n[س4-جـ] الرجوع حسب طريقة الدفع:\n", reorder_by_payment)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
reorder_by_vehicle.plot(kind="bar", ax=axes[0], color="#428bca")
axes[0].set_title("Reorder Rate by Vehicle Type")
axes[0].set_ylabel("Reorder Rate (%)")
reorder_by_cuisine.plot(kind="bar", ax=axes[1], color="#f0ad4e")
axes[1].set_title("Reorder Rate by Cuisine")
reorder_by_payment.plot(kind="bar", ax=axes[2], color="#5bc0de")
axes[2].set_title("Reorder Rate by Payment Method")
plt.tight_layout()
plt.savefig("q4_three_sources.png")
plt.close()

pivot_zone_hour = delivered.pivot_table(
    values="reordered_within_30d", index="zone", columns="order_hour", aggfunc="mean"
)
plt.figure(figsize=(12, 6))
sns.heatmap(pivot_zone_hour, cmap="RdYlGn")
plt.title("Reorder Rate Heatmap: Zone vs Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Zone")
plt.tight_layout()
plt.savefig("q5_zone_hour_heatmap.png")
plt.close()
print("\n[س5] شوف q5_zone_hour_heatmap.png -- دور على المربعات الحمرا (زون+ساعة سيئين مع بعض)")

#some correlation checks
corr_items = delivered["items_count"].corr(delivered["reordered_within_30d"])
print()
print("Q6: Correlation between items_count and reorder rate =", round(corr_items, 3))
print("Close to zero means it has no real effect")
#distance vs delay scatterplot, colored by vehicle type
plt.figure(figsize=(6, 4))
sns.scatterplot(data=delivered, x="distance_km", y="delay_minutes", hue="vehicle_type", alpha=0.5)
plt.title("Delay vs Distance, by Vehicle Type")
plt.xlabel("Distance (km)")
plt.ylabel("Delay (minutes)")
plt.tight_layout()
plt.savefig("q7_confounding_check.png")
plt.close()

corr_dist_delay = delivered["distance_km"].corr(delivered["delay_minutes"])
print("Q7: Correlation between distance and delay =", round(corr_dist_delay, 3))
#checking if missing customer_rating is random or not
rating_missing_delay = delivered.groupby("has_rating")["delay_minutes"].mean()
rating_missing_reorder = delivered.groupby("has_rating")["reordered_within_30d"].mean() * 100

print("\n[تحليل customer_rating المفقود]")
print("متوسط التأخير (عندهم تقييم مقابل من غير تقييم):")
print(rating_missing_delay)
print("معدل الرجوع (عندهم تقييم مقابل من غير تقييم):")
print(rating_missing_reorder)
print(">>> FINDING حقيقي: اللي ملقيمش متأخرين ضعف اللي قيّموا تقريبًا،")
print(">>> وبيرجعوا بمعدل أقل. يعني الغياب مش عشوائي -- ده دليل غير مباشر")
print(">>> على تجربة سيئة اختفت من التقييمات لأن الناس مش بتقيّم لما")
print(">>> تتضايق، تقيّم لما تكون مبسوطة أو عادي.")

# STEP 5 — Part 3: Dashboard (Option A) — 6 panels
# ====================================================================
print("\n" + "=" * 70)
print("STEP 5: Dashboard (Option A) — 6 panels")
print("=" * 70)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
delivered["reordered_within_30d"].value_counts().sort_index().plot(
    kind="bar", ax=axes[0, 0], color=["#d9534f", "#5cb85c"]
)
axes[0, 0].set_title("1) Overall Reorder Rate")
axes[0, 0].set_xlabel("Reordered (0/1)")
axes[0, 0].set_ylabel("Number of Orders")

sns.histplot(delivered["delay_minutes"], bins=40, ax=axes[0, 1], kde=True)
axes[0, 1].axvline(0, color="red", linestyle="--")
axes[0, 1].set_title("2) Delivery Delay Distribution")
axes[0, 1].set_xlabel("Delay (minutes)")

reorder_by_late.plot(kind="bar", ax=axes[0, 2], color=["#5cb85c", "#d9534f"])
axes[0, 2].set_title("3) Reorder Rate: On-time vs Late")
axes[0, 2].set_ylabel("Reorder Rate (%)")

reorder_by_vehicle.plot(kind="bar", ax=axes[1, 0], color="#428bca")
axes[1, 0].set_title("4) Reorder Rate by Vehicle Type")
axes[1, 0].set_ylabel("Reorder Rate (%)")

sns.heatmap(pivot_zone_hour, cmap="RdYlGn", ax=axes[1, 1])
axes[1, 1].set_title("5) Reorder Rate: Zone vs Hour")

sns.scatterplot(
    data=delivered, x="distance_km", y="delay_minutes",
    hue="vehicle_type", alpha=0.5, ax=axes[1, 2]
)
axes[1, 2].set_title("6) Delay vs Distance by Vehicle")

plt.tight_layout()
plt.savefig("sahl_dashboard.png", dpi=150)
plt.close()
print("تم حفظ sahl_dashboard.png")

# ====================================================================
# Part 4 — recommendations (with real numbers from the analysis)
# ====================================================================

print("\n" + "=" * 70)
"""
Recommendation to Nadia:

The data points to two connected problems, not one.

First, delivery reliability is poor: 61.4% of delivered
orders arrived later than promised, and late orders reorder
at only 48% versus 65% for on-time orders. This 17-point
gap is directly tied to a broken promise, not a marketing
problem.

Second, the lateness is not evenly spread: bicycles reorder
at only 41% versus 59% for motorcycles, and delay rises
sharply with distance specifically for bicycles (past 10km,
bicycle delays exceed 75 minutes). This suggests bicycles
are being dispatched on trips too long for them.

Geographically, Abu Qurqas stands out as a consistently weak
zone across most hours of the day, not just a single bad
hour, making it the clearest concentration of the problem.

Recommendation for the 100,000 EGP:
Stop dispatching bicycles for deliveries beyond roughly
8-10km, and prioritize adding motorcycle capacity
specifically in Abu Qurqas during its worst hours. This
targets the exact combination (vehicle + distance + zone)
driving the lowest reorder rates, rather than a general
'improve delivery' initiative.

Dead end:
items_count showed almost no correlation with reordering
(0.02), so basket size is not a meaningful driver and does
not need attention here.

Limitation Nadia should know:
Missing ratings are not random -- unrated orders were
delayed twice as long on average as rated ones. This means
any analysis relying only on customer_rating understates
how bad the real experience was, since dissatisfied
customers are less likely to leave a rating at all.
"""
print("=" * 70)
