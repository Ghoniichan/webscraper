import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# === Load Dataset ===
df = pd.read_csv("linkedin_scraped_job_details_1600.csv")

# === Clean Data ===
df = df.dropna(subset=['title'])
df['title'] = df['title'].str.strip()

# =============================
# 1️⃣ TOPIC CATEGORIZATION (based on description)
# =============================
def categorize_job(desc):
    desc = str(desc).lower()
    if any(word in desc for word in ["developer", "software", "programmer", "engineer", "frontend", "backend", "full stack"]):
        return "Software / IT"
    elif any(word in desc for word in ["data", "ai", "machine learning", "analytics", "scientist"]):
        return "Data / AI / Analytics"
    elif any(word in desc for word in ["marketing", "sales", "advertising", "seo"]):
        return "Marketing / Sales"
    elif any(word in desc for word in ["finance", "accounting", "bank", "audit"]):
        return "Finance / Accounting"
    elif any(word in desc for word in ["hr", "recruit", "human resource", "talent"]):
        return "Human Resources"
    elif any(word in desc for word in ["design", "ui", "ux", "graphics"]):
        return "Design / Creative"
    else:
        return "Other"

df["category"] = df["description"].apply(categorize_job)

topic_counts = df["category"].value_counts().reset_index()
topic_counts.columns = ["Category", "Count"]

fig_topic = px.bar(
    topic_counts,
    x="Category",
    y="Count",
    title="Job Categories Based on Description",
    color="Count",
    text="Count",
    color_continuous_scale="Viridis"
)
fig_topic.update_traces(textposition="outside")
fig_topic.update_layout(xaxis_tickangle=-45)
fig_topic.show()

# =============================
# 2️⃣ TOP 10 MOST COMMON JOB TITLES
# =============================
top_jobs = df['title'].value_counts().head(10).reset_index()
top_jobs.columns = ['Job Title', 'Count']

fig_title = px.bar(
    top_jobs,
    x='Job Title',
    y='Count',
    title='Top 10 Most Common Job Titles',
    color='Count',
    color_continuous_scale='Blues',
    text='Count'
)
fig_title.update_traces(textposition='outside')
fig_title.update_layout(xaxis_tickangle=-45)
fig_title.show()

# =============================
# 3️⃣ TOP 10 HIRING COMPANIES
# =============================
top_companies = df['company'].dropna().value_counts().head(10).reset_index()
top_companies.columns = ['Company', 'Count']

fig_company = px.bar(
    top_companies,
    x='Company',
    y='Count',
    title='Top 10 Hiring Companies',
    color='Count',
    color_continuous_scale='Tealgrn',
    text='Count'
)
fig_company.update_traces(textposition='outside')
fig_company.update_layout(xaxis_tickangle=-45)
fig_company.show()

# =============================
# 4️⃣ JOB DISTRIBUTION BY LOCATION
# =============================
top_locations = df['location'].dropna().value_counts().head(10).reset_index()
top_locations.columns = ['Location', 'Count']

fig_location = px.bar(
    top_locations,
    x='Location',
    y='Count',
    title='Top 10 Locations with Most Job Postings',
    color='Count',
    color_continuous_scale='Oranges',
    text='Count'
)
fig_location.update_traces(textposition='outside')
fig_location.update_layout(xaxis_tickangle=-45)
fig_location.show()

# =============================
# 5️⃣ CORRELATION HEATMAP (if numeric columns exist)
# =============================
numeric_df = df.select_dtypes(include=['float64', 'int64'])
if not numeric_df.empty:
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()
else:
    print("⚠️ No numeric columns found for correlation heatmap.")

print("\n✅ All visualizations completed successfully!")
