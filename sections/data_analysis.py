import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from utils.treat_text import get_pos_and_neg_words


def show(df_merged):
    st.header("Data analysis")

    # Calculate missing data summary
    na_counts = df_merged.isna().sum()
    total_counts = df_merged.shape[0]
    na_percentage = (na_counts / total_counts) * 100

    na_summary = pd.DataFrame({
        'Missing Values': na_counts,
        'Missing %': na_percentage.round(2)
    }).sort_values(by='Missing %', ascending=False)

    st.subheader("Missing Data Overview")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=na_summary['Missing %'],
        y=na_summary.index,
        palette='viridis',
        ax=ax
    )

    # Add raw value labels
    for i, (value, count) in enumerate(zip(na_summary['Missing %'], na_summary['Missing Values'])):
        ax.text(value + 0.5, i, f"{count} missing", va='center')

    ax.set_xlabel("Missing Percentage")
    ax.set_ylabel("Columns")
    ax.set_title("Missing Data by Column")
    st.pyplot(fig)

    st.success("""
Some columns, such as `InternetService`, `SatisfactionScore`, and `AvgDataUsage_GB`, have a high proportion of missing data.

Removing these entries would significantly reduce the dataset and risk distorting the sample. To avoid this, we’ve implemented a targeted strategy for handling missing values, detailed in the , outlined in the [Missing Data Strategy](?section=Missing%20Data%20Strategy) section. This approach preserves data integrity while minimizing bias introduced by deletion.
""")
    
    def classify_columns(df, target, cat_threshold=20, text_threshold=50):
        numeric_cols = []
        categorical_cols = []
        text_cols = []

        for col in df.columns:
            if col in [target, "customerID"]:
                continue

            unique_vals = df[col].nunique(dropna=True)
            dtype = df[col].dtype

            if pd.api.types.is_object_dtype(df[col]):
                max_len = df[col].dropna().astype(str).map(len).max()
                if max_len and max_len > text_threshold:
                    text_cols.append(col)
                else:
                    categorical_cols.append(col)

            elif pd.api.types.is_bool_dtype(df[col]):
                categorical_cols.append(col)

            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                categorical_cols.append(col)

            elif pd.api.types.is_integer_dtype(df[col]):
                if unique_vals <= cat_threshold:
                    categorical_cols.append(col)
                else:
                    numeric_cols.append(col)

            elif pd.api.types.is_float_dtype(df[col]):
                numeric_cols.append(col)

            else:
                categorical_cols.append(col)

        
        return numeric_cols, categorical_cols, text_cols

    target = "Churn"
    numeric_cols, categorical_cols, text_cols = classify_columns(df_merged,target)

    def format_column_list(title: str, columns: list[str]):
        if columns:
            st.markdown(f"**{title} ({len(columns)}):**")
            st.write(", ".join(columns))
        else:
            st.markdown(f"**{title}:** None detected.")

    st.subheader("Variable Typing")


    # Display categorized columns
    format_column_list("Numeric columns", numeric_cols)
    format_column_list("Categorical columns", categorical_cols)
    format_column_list("Text columns", text_cols)

    st.markdown("**Target :**" )
    st.write(target)

    st.markdown("**Identifier :**")
    st.write("customerID")

    st.success("""
Each variable has been classified based on its structure.

This classification ensures that every variable is handled appropriately in the following steps.
    """)



    st.subheader("Target Variable Distribution")

    # Plot churn distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    order = df_merged[target].value_counts().index
    sns.countplot(data=df_merged, x=target, order=order, palette='pastel', ax=ax)

    # Add value labels
    for p in ax.patches:
        count = int(p.get_height())
        ax.annotate(f'{count}', (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom')

    ax.set_title("Churn Distribution")
    ax.set_ylabel("Number of Clients")
    st.pyplot(fig)


    st.markdown(f"""
The number of clients who **did not churn** is about four times higher than those who **did**.  
This imbalance means that churned clients are underrepresented in the data.
    """)

    st.warning("""
If we train a model without addressing this imbalance, it may struggle to correctly identify clients at risk of leaving.  
To improve accuracy, we’ll adjust the way the model learns so it pays equal attention to both groups.
    """)

    st.success(
        """
        Based on our initial data exploration, we plan to approach this as a supervised learning problem. The presence of a clearly defined target column Churn suggests that we can train a model to predict future outcomes using labeled examples.

        Since Churn is a binary variable (indicating whether a client has left or stayed), this points toward a classification task rather than regression, which would require a continuous target.

        We've also observed a significant imbalance in the target distribution: non-churned clients outnumber churned clients by roughly 4 to 1. This imbalance will need to be addressed during model training to ensure fair and accurate predictions. One option we’re considering is applying class weighting to help the model treat both classes with equal importance.
        """
    )




    # Generate positive and negative word sets
    positive_words, negative_words = get_pos_and_neg_words(df_merged)
    

    st.subheader("Text analysis")
    st.markdown("This section presents a naive approach to identifying sentiment-bearing words based solely on their average satisfaction scores. Words are visualized by frequency and colored by score intensity, offering a first glance at potential patterns.")

    # Create word clouds
    def plot_wordcloud(df_words, title, base_color, direction="up"):
        # Size = frequency
        frequencies = {row["token"]: row["count"] for _, row in df_words.iterrows()}
        scores = {row["token"]: row["avg_score"] for _, row in df_words.iterrows()}
        min_score = min(scores.values())
        max_score = max(scores.values())

        # Color function with directional control
        def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            score = scores.get(word, 0)
            norm_score = (score - min_score) / (max_score - min_score + 1e-6)

            if direction == "up":
                intensity = norm_score  # higher score = more vibrant
            elif direction == "down":
                intensity = 1 - norm_score  # lower score = more vibrant
            else:
                intensity = 0.5  # fallback neutral

            if base_color == "green":
                return f"hsl(120, {int(intensity * 100)}%, 40%)"
            elif base_color == "red":
                return f"hsl(0, {int(intensity * 100)}%, 40%)"
            elif base_color == "blue":
                return f"hsl(210, {int(intensity * 100)}%, 40%)"
            else:
                return "black"

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            color_func=color_func
        ).generate_from_frequencies(frequencies)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        ax.set_title(title, fontsize=16)
        st.pyplot(fig)


    # Display positive word cloud
    plot_wordcloud(positive_words, "Words associated with positive rating", "green")

    # Display negative word cloud
    plot_wordcloud(negative_words, "Words associated with negative ratings", "red","down")
    st.warning(
        """
        Our initial text analysis relied on a naive scoring approach based on average satisfaction scores per word. 
        While this surfaced some interesting terms, the results were inconclusive due to noise and ambiguity in the method.
        """
    )

    st.success(
        """
        To address this limitation, we’ve chosen to apply TF-IDF vectorization for text features. 
        This approach captures term relevance more effectively and will be integrated during model training to improve interpretability and predictive power.
        """
    )



    st.subheader("Other Variable Distributions")

    st.markdown("""
We reviewed the distribution of other variables to explore potential patterns or irregularities.

At this stage, no clear insights emerged from these visual checks alone.
    """)

    # Optional: show example plots in expanders
    for col in categorical_cols:
        if col != target:
            with st.expander(f"Distribution of {col}"):
                fig, ax = plt.subplots(figsize=(6, 4))
                order = df_merged[col].value_counts().index
                sns.countplot(data=df_merged, x=col, order=order, palette='pastel', ax=ax)
                ax.set_title(f"{col} (Missing: {df_merged[col].isna().sum()})")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                st.pyplot(fig)

    for col in numeric_cols:
        with st.expander(f"Distribution of {col}"):
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df_merged[col].dropna(), kde=True, ax=ax, color='skyblue')
            ax.set_title(f"{col} (Missing: {df_merged[col].isna().sum()})")
            st.pyplot(fig)

    st.subheader("Bivariate Analysis")

    st.markdown("""
We explored how each variable behaves depending on whether a client has churned or not.  
This helps identify potential differences between the two groups.
    """)

    # Display numeric comparisons
    for col in numeric_cols:
        with st.expander(f"{col} by Churn"):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Histogram + KDE
            sns.histplot(data=df_merged, x=col, hue=target, kde=True, bins=30,
                        palette='Set2', element='step', ax=axes[0])
            axes[0].set_title(f'Distribution of {col} by {target}')

            # Boxplot
            sns.boxplot(data=df_merged, x=target, y=col, palette='Set3', ax=axes[1])
            axes[1].set_title(f'Boxplot of {col} by {target}')

            st.pyplot(fig)

    # Display categorical comparisons
    for col in categorical_cols:
        with st.expander(f"{col} by Churn (Counts)"):
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.countplot(data=df_merged, x=col, hue=target, palette='pastel', ax=ax)
            ax.set_title(f'{col} distribution by {target} (Counts)')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            st.pyplot(fig)

        with st.expander(f"{col} by Churn (Normalized)"):
            # Create normalized proportions safely
            grouped = df_merged.groupby([col, target]).size()
            normalized = grouped / grouped.groupby(level=0).sum()
            prop_df = normalized.reset_index().rename(columns={0: 'proportion'})

            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(data=prop_df, x=col, y='proportion', hue=target, palette='pastel', ax=ax)
            ax.set_title(f'{col} distribution by {target} (Normalized)')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            st.pyplot(fig)




    st.success(
        "Some variables, such as `tenure`, show clear differences in their distribution shapes between churned and non-churned clients.  \n" 
        "This suggests they may have a meaningful influence on churn behavior.\n\n"
        "Other variables appear more evenly distributed across both groups, indicating they may be less relevant for predictive modeling"
    )

    st.subheader("Pairwise Variable Analysis")

    st.markdown("""
    We also reviewed how numeric variables interact with each other, in relation to churn.  
    This helps detect potential combinations or patterns that may not be visible in single-variable views.
    """)

    with st.expander("Pairplot of Numeric Variables by Churn"):
        fig = sns.pairplot(
            data=df_merged[numeric_cols + [target]].dropna(),
            hue=target,
            diag_kind="kde",
            corner=True,
            plot_kws={"alpha": 0.4, "s": 15}
        )
        st.pyplot(fig)

    st.markdown("""
    No strong or consistent patterns were observed in this view.  
    """)


    st.subheader("Correlation Analysis")

    st.markdown("""
We examined correlations between numeric variables to identify potential redundancies — cases where one variable could be explained by a combination of others.
    """)

    with st.expander("Correlation Matrix (Numeric Features)"):
        df_numeric = df_merged[numeric_cols]
        cor_matrix = df_numeric.corr()
        mask = np.triu(np.ones_like(cor_matrix, dtype=bool))

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cor_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", square=True,
                    cbar_kws={"shrink": 0.8}, linewidths=0.5, linecolor='gray', ax=ax)
        ax.set_title("Lower Triangle Correlation Matrix (Numeric Features)")
        st.pyplot(fig)

    st.markdown("""
No strong correlations or anti-correlations were found, the highest observed value was **0.49**.  
    """)

    st.success("This suggests that the numeric variables are relatively independent and none appear to be simple combinations of others. This supports their inclusion in the model without immediate need for dimensionality reduction.")