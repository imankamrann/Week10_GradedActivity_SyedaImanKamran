# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import os
# from io import StringIO

# # ========================================
# # PAGE CONFIG
# # ========================================
# st.set_page_config(
#     page_title="Resource Tagging Dashboard",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # ========================================
# # TITLE
# # ========================================
# st.title("Resource Tagging: Cost Governance Simulator")
# st.markdown("**INFO 49971 – Cloud Economics | Week 10 Activity**")

# # ========================================
# # DATA LOADING
# # ========================================
# data_folder = "data"

# if not os.path.exists(data_folder):
#     st.error(f"Folder '{data_folder}' not found. Please create it and add the CSV file.")
#     st.stop()

# # --- Load Dataset ---
# dataset_path = os.path.join(data_folder, "cloudmart_multi_account.csv")
# if not os.path.exists(dataset_path):
#     st.error(f"Dataset file not found: {dataset_path}")
#     st.stop()

# # Read and clean the CSV to handle potential quoted lines
# with open(dataset_path, 'r') as f:
#     lines = f.readlines()

# clean_lines = [line.strip().strip('"') for line in lines]
# csv_string = '\n'.join(clean_lines)
# df = pd.read_csv(StringIO(csv_string))
# df = df.drop_duplicates()  # Remove duplicates as the data repeats

# st.success(f"Loaded dataset ({len(df):,} rows).")

# # ========================================
# # DATA CLEANING
# # ========================================
# # Handle missing values - fill with 'Unknown' for tags
# tag_columns = ['Department', 'Project', 'Environment', 'Owner', 'CostCenter']
# for col in tag_columns:
#     df[col] = df[col].fillna('Unknown')

# df['MonthlyCostUSD'] = df['MonthlyCostUSD'].fillna(0)

# # Calculate Tag Completeness Score
# df['TagScore'] = df[tag_columns].apply(lambda row: (row != 'Unknown').sum(), axis=1)

# # ========================================
# # SIDEBAR FILTERS
# # ========================================
# st.sidebar.header("Filters")

# services = st.sidebar.multiselect("Service", df["Service"].unique())
# regions = st.sidebar.multiselect("Region", df["Region"].unique())
# departments = st.sidebar.multiselect("Department", df["Department"].unique())
# environments = st.sidebar.multiselect("Environment", df["Environment"].unique())

# # Apply Filters
# filtered_df = df[
#     (df["Service"].isin(services) if services else True) &
#     (df["Region"].isin(regions) if regions else True) &
#     (df["Department"].isin(departments) if departments else True) &
#     (df["Environment"].isin(environments) if environments else True)
# ].copy()

# # ========================================
# # COMPUTATIONS
# # ========================================
# total_resources = len(filtered_df)
# tagged_count = filtered_df['Tagged'].value_counts().get('Yes', 0)
# untagged_count = filtered_df['Tagged'].value_counts().get('No', 0)
# untagged_percent = (untagged_count / total_resources * 100) if total_resources > 0 else 0

# total_cost = filtered_df['MonthlyCostUSD'].sum()
# tagged_cost = filtered_df[filtered_df['Tagged'] == 'Yes']['MonthlyCostUSD'].sum()
# untagged_cost = filtered_df[filtered_df['Tagged'] == 'No']['MonthlyCostUSD'].sum()
# untagged_cost_percent = (untagged_cost / total_cost * 100) if total_cost > 0 else 0

# dept_untagged_cost = filtered_df[filtered_df['Tagged'] == 'No'].groupby('Department')['MonthlyCostUSD'].sum().sort_values(ascending=False)
# top_project_cost = filtered_df.groupby('Project')['MonthlyCostUSD'].sum().sort_values(ascending=False).head(1)

# env_comparison = filtered_df.groupby(['Environment', 'Tagged'])['MonthlyCostUSD'].sum().reset_index()

# top5_lowest_tags = filtered_df.sort_values('TagScore').head(5)
# most_missing_tags = df[tag_columns].apply(lambda col: (col == 'Unknown').sum()).sort_values(ascending=False)

# untagged_df = filtered_df[filtered_df['Tagged'] == 'No']

# # ========================================
# # KPI CARDS
# # ========================================
# col1, col2, col3, col4 = st.columns(4)
# col1.metric("Total Resources", total_resources)
# col2.metric("Untagged %", f"{untagged_percent:.1f}%")
# col3.metric("Total Cost (USD)", f"${total_cost:.2f}")
# col4.metric("Untagged Cost (USD)", f"${untagged_cost:.2f}")

# # ========================================
# # TABS
# # ========================================
# tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Exploration", "Cost Visibility", "Tagging Compliance", "Visualization Dashboard", "Tag Remediation"])

# # ---------- TAB 1: Data Exploration ----------
# with tab1:
#     st.subheader("Dataset Overview")
#     st.write("**First 5 Rows**")
#     st.dataframe(filtered_df.head())

#     st.write("**Missing Values**")
#     st.write(filtered_df.isnull().sum())

#     st.write("**Columns with Most Missing Values**")
#     st.write(most_missing_tags)

#     st.write("**Tagged vs Untagged Count**")
#     st.write(filtered_df['Tagged'].value_counts())

#     st.write("**Untagged Percentage**")
#     st.write(f"{untagged_percent:.1f}%")

# # ---------- TAB 2: Cost Visibility ----------
# with tab2:
#     st.subheader("Cost Analysis")
#     st.write("**Tagged vs Untagged Costs**")
#     cost_by_tag = filtered_df.groupby('Tagged')['MonthlyCostUSD'].sum().reset_index()
#     st.dataframe(cost_by_tag)

#     st.write("**Untagged Cost Percentage**")
#     st.write(f"{untagged_cost_percent:.1f}%")

#     st.write("**Department with Most Untagged Cost**")
#     st.dataframe(dept_untagged_cost.reset_index())

#     st.write("**Project with Most Cost**")
#     st.dataframe(top_project_cost.reset_index())

#     st.write("**Prod vs Dev Comparison**")
#     st.dataframe(env_comparison)

# # ---------- TAB 3: Tagging Compliance ----------
# with tab3:
#     st.subheader("Compliance Metrics")
#     st.write("**Tag Completeness Score (Sample)**")
#     st.dataframe(filtered_df[['ResourceID', 'TagScore']].head())

#     st.write("**Top 5 Resources with Lowest Completeness**")
#     st.dataframe(top5_lowest_tags[['ResourceID', 'Service', 'TagScore']])

#     st.write("**Most Frequently Missing Tags**")
#     st.dataframe(most_missing_tags.reset_index())

#     st.write("**Untagged Resources**")
#     st.dataframe(untagged_df)

#     st.download_button("Export Untagged to CSV", untagged_df.to_csv(index=False).encode(), "untagged.csv", "text/csv")

# # ---------- TAB 4: Visualization Dashboard ----------
# with tab4:
#     st.subheader("Visual Insights")

#     col1, col2 = st.columns(2)
#     with col1:
#         st.write("**Tagged vs Untagged Resources**")
#         fig_pie = px.pie(filtered_df, names='Tagged', title="Tagged Status")
#         st.plotly_chart(fig_pie, use_container_width=True)

#     with col2:
#         st.write("**Cost per Department by Tagging**")
#         fig_bar_dept = px.bar(filtered_df, x='Department', y='MonthlyCostUSD', color='Tagged', barmode='group', title="Cost by Department")
#         st.plotly_chart(fig_bar_dept, use_container_width=True)

#     st.write("**Total Cost per Service**")
#     cost_by_service = filtered_df.groupby('Service')['MonthlyCostUSD'].sum().reset_index()
#     fig_bar_service = px.bar(cost_by_service, y='Service', x='MonthlyCostUSD', orientation='h', title="Cost by Service")
#     st.plotly_chart(fig_bar_service, use_container_width=True)

#     st.write("**Cost by Environment**")
#     cost_by_env = filtered_df.groupby('Environment')['MonthlyCostUSD'].sum().reset_index()
#     fig_pie_env = px.pie(cost_by_env, names='Environment', values='MonthlyCostUSD', title="Cost by Environment")
#     st.plotly_chart(fig_pie_env, use_container_width=True)

# # ---------- TAB 5: Tag Remediation ----------
# with tab5:
#     st.subheader("Remediation Workflow")
#     st.write("Edit untagged resources below. Fill in missing tags.")

#     if 'edited_df' not in st.session_state:
#         st.session_state.edited_df = untagged_df.copy()

#     edited_df = st.data_editor(
#         st.session_state.edited_df,
#         column_config={
#             "Department": st.column_config.TextColumn(),
#             "Project": st.column_config.TextColumn(),
#             "Environment": st.column_config.TextColumn(),
#             "Owner": st.column_config.TextColumn(),
#             "CostCenter": st.column_config.TextColumn(),
#         },
#         num_rows="dynamic"
#     )

#     if st.button("Update and Recalculate"):
#         # Simulate updating Tagged to Yes if tags filled
#         edited_df['Tagged'] = edited_df[tag_columns].apply(lambda row: 'Yes' if all(row != 'Unknown') else 'No', axis=1)
#         st.session_state.edited_df = edited_df

#         # Recalculate metrics
#         new_untagged_count = (edited_df['Tagged'] == 'No').sum()
#         new_untagged_percent = (new_untagged_count / len(edited_df) * 100) if len(edited_df) > 0 else 0
#         new_untagged_cost = edited_df[edited_df['Tagged'] == 'No']['MonthlyCostUSD'].sum()
#         st.write("**After Remediation:**")
#         st.write(f"Untagged Resources: {new_untagged_count}")
#         st.write(f"Untagged %: {new_untagged_percent:.1f}%")
#         st.write(f"Untagged Cost: ${new_untagged_cost:.2f}")

#     st.download_button("Download Remediated CSV", edited_df.to_csv(index=False).encode(), "remediated.csv", "text/csv")

#     st.markdown("### Reflection")
#     st.write("Improved tagging enhances accountability by attributing costs to specific departments and projects. It improves reporting accuracy and helps in better governance by identifying and reducing hidden costs.")

# # ========================================
# # DOWNLOAD BUTTONS
# # ========================================
# st.sidebar.markdown("---")
# st.sidebar.download_button("Download Original Data", df.to_csv(index=False).encode(), "original.csv", "text/csv")
# st.sidebar.download_button("Download Filtered Data", filtered_df.to_csv(index=False).encode(), "filtered.csv", "text/csv")

# # ========================================
# # FOOTER
# # ========================================
# st.markdown("---")
# st.caption("Built with Streamlit • Fall 2025 • Sheridan College")


# empty filters usecase 

import streamlit as st
import pandas as pd
import plotly.express as px
import os
from io import StringIO

# ========================================
# PAGE CONFIG
# ========================================
st.set_page_config(
    page_title="Resource Tagging Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# TITLE
# ========================================
st.title("Resource Tagging: Cost Governance Simulator")
st.markdown("**INFO 49971 – Cloud Economics | Week 10 Activity**")

# ========================================
# DATA LOADING
# ========================================
data_folder = "data"

if not os.path.exists(data_folder):
    st.error(f"Folder '{data_folder}' not found. Please create it and add the CSV file.")
    st.stop()

# --- Load Dataset ---
dataset_path = os.path.join(data_folder, "cloudmart_multi_account.csv")
if not os.path.exists(dataset_path):
    st.error(f"Dataset file not found: {dataset_path}")
    st.stop()

# Read and clean the CSV to handle potential quoted lines
with open(dataset_path, 'r') as f:
    lines = f.readlines()

clean_lines = [line.strip().strip('"') for line in lines]
csv_string = '\n'.join(clean_lines)
df = pd.read_csv(StringIO(csv_string))
df = df.drop_duplicates()  # Remove duplicates as the data repeats

st.success(f"Loaded dataset ({len(df):,} rows).")

# ========================================
# DATA CLEANING
# ========================================
# Handle missing values - fill with 'Unknown' for tags
tag_columns = ['Department', 'Project', 'Environment', 'Owner', 'CostCenter']
for col in tag_columns:
    df[col] = df[col].fillna('Unknown')

df['MonthlyCostUSD'] = df['MonthlyCostUSD'].fillna(0)

# Calculate Tag Completeness Score
df['TagScore'] = df[tag_columns].apply(lambda row: (row != 'Unknown').sum(), axis=1)

# ========================================
# SIDEBAR FILTERS
# ========================================
st.sidebar.header("Filters")

services = st.sidebar.multiselect("Service", df["Service"].unique())
regions = st.sidebar.multiselect("Region", df["Region"].unique())
departments = st.sidebar.multiselect("Department", df["Department"].unique())
environments = st.sidebar.multiselect("Environment", df["Environment"].unique())

# Apply Filters
mask = pd.Series([True] * len(df))
if services:
    mask &= df["Service"].isin(services)
if regions:
    mask &= df["Region"].isin(regions)
if departments:
    mask &= df["Department"].isin(departments)
if environments:
    mask &= df["Environment"].isin(environments)
filtered_df = df[mask].copy()

# ========================================
# COMPUTATIONS
# ========================================
total_resources = len(filtered_df)
tagged_count = filtered_df['Tagged'].value_counts().get('Yes', 0)
untagged_count = filtered_df['Tagged'].value_counts().get('No', 0)
untagged_percent = (untagged_count / total_resources * 100) if total_resources > 0 else 0

total_cost = filtered_df['MonthlyCostUSD'].sum()
tagged_cost = filtered_df[filtered_df['Tagged'] == 'Yes']['MonthlyCostUSD'].sum()
untagged_cost = filtered_df[filtered_df['Tagged'] == 'No']['MonthlyCostUSD'].sum()
untagged_cost_percent = (untagged_cost / total_cost * 100) if total_cost > 0 else 0

dept_untagged_cost = filtered_df[filtered_df['Tagged'] == 'No'].groupby('Department')['MonthlyCostUSD'].sum().sort_values(ascending=False)
top_project_cost = filtered_df.groupby('Project')['MonthlyCostUSD'].sum().sort_values(ascending=False).head(1)

env_comparison = filtered_df.groupby(['Environment', 'Tagged'])['MonthlyCostUSD'].sum().reset_index()

top5_lowest_tags = filtered_df.sort_values('TagScore').head(5)
most_missing_tags = df[tag_columns].apply(lambda col: (col == 'Unknown').sum()).sort_values(ascending=False)

untagged_df = filtered_df[filtered_df['Tagged'] == 'No']

# ========================================
# KPI CARDS
# ========================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Resources", total_resources)
col2.metric("Untagged %", f"{untagged_percent:.1f}%")
col3.metric("Total Cost (USD)", f"${total_cost:.2f}")
col4.metric("Untagged Cost (USD)", f"${untagged_cost:.2f}")

# ========================================
# TABS
# ========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Exploration", "Cost Visibility", "Tagging Compliance", "Visualization Dashboard", "Tag Remediation"])

# ---------- TAB 1: Data Exploration ----------
with tab1:
    st.subheader("Dataset Overview")
    st.write("**First 5 Rows**")
    st.dataframe(filtered_df.head())

    st.write("**Missing Values**")
    st.write(filtered_df.isnull().sum())

    st.write("**Columns with Most Missing Values**")
    st.write(most_missing_tags)

    st.write("**Tagged vs Untagged Count**")
    st.write(filtered_df['Tagged'].value_counts())

    st.write("**Untagged Percentage**")
    st.write(f"{untagged_percent:.1f}%")

# ---------- TAB 2: Cost Visibility ----------
with tab2:
    st.subheader("Cost Analysis")
    st.write("**Tagged vs Untagged Costs**")
    cost_by_tag = filtered_df.groupby('Tagged')['MonthlyCostUSD'].sum().reset_index()
    st.dataframe(cost_by_tag)

    st.write("**Untagged Cost Percentage**")
    st.write(f"{untagged_cost_percent:.1f}%")

    st.write("**Department with Most Untagged Cost**")
    st.dataframe(dept_untagged_cost.reset_index())

    st.write("**Project with Most Cost**")
    st.dataframe(top_project_cost.reset_index())

    st.write("**Prod vs Dev Comparison**")
    st.dataframe(env_comparison)

# ---------- TAB 3: Tagging Compliance ----------
with tab3:
    st.subheader("Compliance Metrics")
    st.write("**Tag Completeness Score (Sample)**")
    st.dataframe(filtered_df[['ResourceID', 'TagScore']].head())

    st.write("**Top 5 Resources with Lowest Completeness**")
    st.dataframe(top5_lowest_tags[['ResourceID', 'Service', 'TagScore']])

    st.write("**Most Frequently Missing Tags**")
    st.dataframe(most_missing_tags.reset_index())

    st.write("**Untagged Resources**")
    st.dataframe(untagged_df)

    st.download_button("Export Untagged to CSV", untagged_df.to_csv(index=False).encode(), "untagged.csv", "text/csv")

# ---------- TAB 4: Visualization Dashboard ----------
with tab4:
    st.subheader("Visual Insights")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Tagged vs Untagged Resources**")
        fig_pie = px.pie(filtered_df, names='Tagged', title="Tagged Status")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.write("**Cost per Department by Tagging**")
        fig_bar_dept = px.bar(filtered_df, x='Department', y='MonthlyCostUSD', color='Tagged', barmode='group', title="Cost by Department")
        st.plotly_chart(fig_bar_dept, use_container_width=True)

    st.write("**Total Cost per Service**")
    cost_by_service = filtered_df.groupby('Service')['MonthlyCostUSD'].sum().reset_index()
    fig_bar_service = px.bar(cost_by_service, y='Service', x='MonthlyCostUSD', orientation='h', title="Cost by Service")
    st.plotly_chart(fig_bar_service, use_container_width=True)

    st.write("**Cost by Environment**")
    cost_by_env = filtered_df.groupby('Environment')['MonthlyCostUSD'].sum().reset_index()
    fig_pie_env = px.pie(cost_by_env, names='Environment', values='MonthlyCostUSD', title="Cost by Environment")
    st.plotly_chart(fig_pie_env, use_container_width=True)

# ---------- TAB 5: Tag Remediation ----------
with tab5:
    st.subheader("Remediation Workflow")
    st.write("Edit untagged resources below. Fill in missing tags.")

    if 'edited_df' not in st.session_state:
        st.session_state.edited_df = untagged_df.copy()

    edited_df = st.data_editor(
        st.session_state.edited_df,
        column_config={
            "Department": st.column_config.TextColumn(),
            "Project": st.column_config.TextColumn(),
            "Environment": st.column_config.TextColumn(),
            "Owner": st.column_config.TextColumn(),
            "CostCenter": st.column_config.TextColumn(),
        },
        num_rows="dynamic"
    )

    if st.button("Update and Recalculate"):
        # Simulate updating Tagged to Yes if tags filled
        edited_df['Tagged'] = edited_df[tag_columns].apply(lambda row: 'Yes' if all(row != 'Unknown') else 'No', axis=1)
        st.session_state.edited_df = edited_df

        # Recalculate metrics
        new_untagged_count = (edited_df['Tagged'] == 'No').sum()
        new_untagged_percent = (new_untagged_count / len(edited_df) * 100) if len(edited_df) > 0 else 0
        new_untagged_cost = edited_df[edited_df['Tagged'] == 'No']['MonthlyCostUSD'].sum()
        st.write("**After Remediation:**")
        st.write(f"Untagged Resources: {new_untagged_count}")
        st.write(f"Untagged %: {new_untagged_percent:.1f}%")
        st.write(f"Untagged Cost: ${new_untagged_cost:.2f}")

    st.download_button("Download Remediated CSV", edited_df.to_csv(index=False).encode(), "remediated.csv", "text/csv")

    st.markdown("### Reflection")
    st.write("Improved tagging enhances accountability by attributing costs to specific departments and projects. It improves reporting accuracy and helps in better governance by identifying and reducing hidden costs.")

# ========================================
# DOWNLOAD BUTTONS
# ========================================
st.sidebar.markdown("---")
st.sidebar.download_button("Download Original Data", df.to_csv(index=False).encode(), "original.csv", "text/csv")
st.sidebar.download_button("Download Filtered Data", filtered_df.to_csv(index=False).encode(), "filtered.csv", "text/csv")

# ========================================
# FOOTER
# ========================================
st.markdown("---")
st.caption("Built with Streamlit • Fall 2025 • Sheridan College")