import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from wordcloud import WordCloud

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer


from utils.imputation import CustomImputer, TextCleaner

import re
    

    # ============================
    #  Model Trials (Internal Use Only)
    # ============================

    # These were initial experiments to compare different classifiers.
    # They helped identify Random Forest as the best performer for this use case.

    # from sklearn.linear_model import LogisticRegression
    # from sklearn.ensemble import GradientBoostingClassifier
    # from sklearn.metrics import confusion_matrix
    # import seaborn as sns

    # models = {
    #     "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42),
    #     "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, class_weight="balanced", random_state=42),
    #     "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    # }

    # trained_pipelines = {}

    # def scan_for_str_int_mixture(df):
    #     for col in df.columns:
    #         str_like = df[col].apply(lambda x: isinstance(x, str) and x.strip().isdigit())
    #         int_like = df[col].apply(lambda x: isinstance(x, int))
    #         if str_like.any() and int_like.any():
    #             print(f"Column '{col}' has both str-digit and int values")

    # scan_for_str_int_mixture(df_merged)

    # for name, model in models.items():
    #     print(f"\n{name} - Training")
    #     pipeline = Pipeline([
    #         ("custom_imputer", CustomImputer()),
    #         ("preprocess", preprocessor),
    #         ("classifier", model)
    #     ])
    #     pipeline.fit(X_train, y_train)
    #     trained_pipelines[name] = pipeline

    # for name, pipeline in trained_pipelines.items():
    #     print(f"\n{name} - Final Evaluation")
    #     y_test_pred = pipeline.predict(X_test)
    #     y_test_proba = pipeline.predict_proba(X_test)[:, 1]

    #     cm = confusion_matrix(y_test, y_test_pred, labels=pipeline.named_steps["classifier"].classes_)
    #     plt.figure(figsize=(6, 4))
    #     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
    #                 xticklabels=pipeline.named_steps["classifier"].classes_,
    #                 yticklabels=pipeline.named_steps["classifier"].classes_)
    #     plt.xlabel("Predicted")
    #     plt.ylabel("Actual")
    #     plt.title(f"Confusion Matrix - {name}")
    #     plt.tight_layout()
    #     plt.show()

    #     print("Test Set Classification Report:")
    #     print(classification_report(y_test, y_test_pred, digits=4))
    #     print(f"PR AUC (Average Precision): {average_precision_score(y_test, y_test_proba, pos_label='Yes'):.4f}")

    #   rf_pipeline = Pipeline([
    #       ("preprocess", preprocessor),
    #       ("classifier", RandomForestClassifier(random_state=42))
    #   ])
    #   
    #   
    #   param_dist = {
    #       "classifier__n_estimators": [100, 200, 300],
    #       "classifier__max_depth": [3, 5, 10, None],
    #       "classifier__min_samples_split": [2, 5, 10],
    #       "classifier__min_samples_leaf": [1, 2, 4],
    #       "classifier__max_features": ["sqrt", "log2", None],
    #       "classifier__class_weight": [None, "balanced", {0: 1, 1: 2}, {0: 1, 1: 3}]
    #   }
    #   
    #   
    #   scorer = make_scorer(average_precision_score, response_method="predict_proba")
    #   
    #   
    #   search = RandomizedSearchCV(
    #       rf_pipeline,
    #       param_distributions=param_dist,
    #       n_iter=30,
    #       scoring=scorer,
    #       cv=5,
    #       verbose=1,
    #       random_state=42,
    #       n_jobs=-1
    #   )
    #   
    #   search.fit(X_train, y_train_encoded)
    #   best_rf_pipeline = search.best_estimator_
    #
    #   y_pred = best_rf_pipeline.predict(X_test)
    #   y_proba = best_rf_pipeline.predict_proba(X_test)[:, 1]
    #   
    #   
    #   print("Default Threshold Classification Report:")
    #   print(classification_report(y_test_encoded, y_pred, target_names=le.classes_))
    #   print("PR AUC (Average Precision):", average_precision_score(y_test_encoded, y_proba))
    #   
    #   
    #   precision, recall, thresholds = precision_recall_curve(y_test_encoded, y_proba)
    #   f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    #   best_f1_idx = f1_scores.argmax()
    #   best_f1_threshold = thresholds[best_f1_idx]
    #   y_pred_f1 = (y_proba >= best_f1_threshold).astype(int)
    #   
    #   print(f"\nBest Threshold by F1: {best_f1_threshold:.3f}")
    #   print("F1-Optimized Classification Report:")
    #   print(classification_report(y_test_encoded, y_pred_f1, target_names=le.classes_))
    #   
    #   
    #   fbeta_scores = [fbeta_score(y_test_encoded, (y_proba >= t).astype(int), beta=2) for t in thresholds]
    #   best_f2_idx = max(range(len(fbeta_scores)), key=lambda i: fbeta_scores[i])
    #   best_f2_threshold = thresholds[best_f2_idx]
    #   y_pred_f2 = (y_proba >= best_f2_threshold).astype(int)
    #   
    #   print(f"\nBest Threshold by F2: {best_f2_threshold:.3f}")
    #   print("F2-Optimized Classification Report:")
    #   print(classification_report(y_test_encoded, y_pred_f2, target_names=le.classes_))
    #   
    #   
    #   precision_yes = precision_score(y_test_encoded, y_pred_f2, pos_label=1)
    #   recall_yes = recall_score(y_test_encoded, y_pred_f2, pos_label=1)
    #   
    #   print(f"\nFinal Detection Metrics at F2 Threshold:")
    #   print(f"Recall (Yes): {recall_yes:.3f}")
    #   print(f"Precision (Yes): {precision_yes:.3f}")
    #   print(f"Threshold Used: {best_f2_threshold:.3f}")
    #   
    #   
    #   plt.figure(figsize=(8, 6))
    #   plt.plot(recall, precision, label="Precision-Recall Curve", color="blue")
    #   plt.axvline(x=recall_yes, color="red", linestyle="--", label=f"Recall @ F2 = {recall_yes:.2f}")
    #   plt.axhline(y=precision_yes, color="green", linestyle="--", label=f"Precision @ F2 = {precision_yes:.2f}")
    #   plt.xlabel("Recall")
    #   plt.ylabel("Precision")
    #   plt.title("Detection Tradeoff for Churners")
    #   plt.legend()
    #   plt.grid(True)
    #   plt.tight_layout()
    #   plt.show()
    #   
    #   print("Best Parameters Found:")
    #   print(search.best_params_)

@st.cache_resource
def train_model(X_train, y_train_encoded, _preprocessor):
    rf_pipeline = Pipeline([
        ("custom_imputer", CustomImputer()),
        ("preprocess", _preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            min_samples_split=5,
            min_samples_leaf=1,
            max_features="log2",
            max_depth=10,
            class_weight= "balanced",
            random_state=42
        ))
    ])
    rf_pipeline.fit(X_train, y_train_encoded)
    return rf_pipeline

def show(df_merged):
    st.header("Customer Churn Detection : Model")
    with st.expander("How This Model Was Chosen"):
        st.markdown("""
        Before selecting the final model, we tested several options to see which one worked best for predicting customer churn.  
        These included simpler models like Logistic Regression and more advanced ones like Gradient Boosting.

        After comparing their results, we selected a Random Forest model. It provided the best balance between identifying clients at risk and avoiding false alerts.

        To fine-tune the model, we used a method called `RandomizedSearchCV`.  
        This approach tries out many different combinations of model settings and evaluates how well each one performs.  
        Instead of testing every possible option (which would take too long), it samples a wide range efficiently and selects the best-performing setup.  
        This helped us find a version of the model that works reliably across different types of clients.

        The final model includes:
        - Careful handling of missing data  
        - Preprocessing for numbers, categories, and text  
        - A balancing step to ensure both churned and non-churned clients are treated fairly  
        - A custom threshold that adjusts predictions based on your business priorities

        These steps ensure the model is accurate, fair, and aligned with how you want to use it.
        """)


    st.markdown("""
    ### Define Your Business Priorities  
    Use the sliders below to guide the model based on what matters most:
    - **Customer Coverage**: What percentage of likely churners should we identify?
    - **Alert Accuracy**: What percentage of alerts should be correct?
    """)

    col1, col2 = st.columns(2)
    with col1:
        min_recall_pct = st.slider("Minimum Customer Coverage (%)", 0, 100, 10, step=5)
        min_recall = min_recall_pct / 100.0
    with col2:
        min_precision_pct = st.slider("Minimum Alert Accuracy (%)", 0, 100, 10, step=5)
        min_precision = min_precision_pct / 100.0

    df_aggregated = df_merged.groupby("customerID", as_index=False, observed=False).agg({
        col: "median" if df_merged[col].dtype.kind in "iufc" else "first"
        for col in df_merged.columns
    })

    X = df_aggregated.drop("Churn", axis=1)
    y = df_aggregated["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    def classify_columns(df, target, cat_threshold=0, text_threshold=50):
        numeric_cols = []
        categorical_cols = []
        text_cols = []

        for col in df.columns:
            if col in [target, "customerID"]:
                continue

            unique_vals = df[col].nunique(dropna=True)

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



    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    text_transformer = Pipeline([
        ('cleaner', TextCleaner()),
        ('tfidf', TfidfVectorizer(
            stop_words='english',
            token_pattern=r'\b[a-zA-Z]{2,}\b'
        ))
    ])


    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
        ("text", text_transformer, text_cols[0])  # TfidfVectorizer accepte une seule colonne à la fois
    ])


    best_rf_pipeline = train_model(X_train, y_train_encoded, preprocessor)
    y_proba = best_rf_pipeline.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test_encoded, y_proba)

    valid_indices = [
        i for i, t in enumerate(thresholds)
        if recall[i] >= min_recall and precision[i] >= min_precision
    ]

    if not valid_indices:
        st.warning("No threshold meets both minimum coverage and accuracy.")
        default_pred = (y_proba >= 0.5).astype(int)
        with st.expander("Default Model Performance (Threshold = 0.50)"):
            st.text(classification_report(
                y_test_encoded,
                default_pred,
                target_names=le.classes_,
                zero_division=0
            ))
        return

    f1_scores = [2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) for i in valid_indices]
    best_idx = valid_indices[np.argmax(f1_scores)]
    chosen_threshold = thresholds[best_idx]
    y_pred_custom = (y_proba >= chosen_threshold).astype(int)

    total_churners = sum(y_test_encoded)
    predicted_churners = sum(y_pred_custom)
    true_positives = sum((y_pred_custom == 1) & (y_test_encoded == 1))

    st.subheader("Model Performance Based on Your Priorities")
    st.write(f"**Threshold used:** `{chosen_threshold:.3f}`")
    st.metric("Churners Detected", f"{true_positives} of {total_churners}")
    st.metric("Customers Alerted", f"{predicted_churners}")
    st.metric("Coverage (Recall)", f"{recall[best_idx]*100:.1f}%")
    st.metric("Accuracy (Precision)", f"{precision[best_idx]*100:.1f}%")

    st.success(f"""
    **Model Summary Based on Your Priorities**  
    - Churner detection rate: **{recall[best_idx]*100:.1f}%**  
    - Alert precision: **{precision[best_idx]*100:.1f}%**  

    This balance reflects your selected thresholds and helps align predictions with operational goals.
    """)

    report_dict = classification_report(
        y_test_encoded,
        y_pred_custom,
        target_names=le.classes_,
        zero_division=0,
        output_dict=True
    )

    report_df = pd.DataFrame(report_dict).transpose()

    y_proba_train = best_rf_pipeline.predict_proba(X_train)[:, 1]
    y_pred_train_custom = (y_proba_train >= chosen_threshold).astype(int)


    train_report_dict = classification_report(
        y_train_encoded,
        y_pred_train_custom,
        target_names=le.classes_,
        zero_division=0,
        output_dict=True
    )
    train_report_df = pd.DataFrame(train_report_dict).transpose()

    with st.expander("Detailed Classification Report"):
        st.markdown("train")
        st.dataframe(train_report_df.style.format("{:.2f}"))
        st.markdown("test")
        st.dataframe(report_df.style.format("{:.2f}"))

    # Generate and display confusion matrix
    cm = confusion_matrix(y_test_encoded, y_pred_custom)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)

    with st.expander("Confusion Matrix"):
        fig, ax = plt.subplots()
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

    st.markdown("""
    ### Detection Tradeoff Curve  
    This chart shows how well our model balances coverage and accuracy.
    """)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, label="Precision-Recall Curve", color="blue")
    ax.axvline(x=recall[best_idx], color="red", linestyle="--", label=f"Coverage @ {recall[best_idx]:.2f}")
    ax.axhline(y=precision[best_idx], color="green", linestyle="--", label=f"Accuracy @ {precision[best_idx]:.2f}")
    ax.set_xlabel("Coverage (Recall)")
    ax.set_ylabel("Accuracy (Precision)")
    ax.set_title("Detection Tradeoff for Churners")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    


    st.header("Customer Churn Detection : Model Analysis")

    ## SHAP
    # Ensure customerID is preserved and cleaned
    X_test = X_test.copy()
    X_test["customerID"] = X_test["customerID"].astype(str).str.strip()

    # Transform features using trained pipeline
    X_test_transformed = best_rf_pipeline.named_steps["preprocess"].transform(X_test)

    # Get feature names from pipeline
    def get_feature_names_custom(preprocessor, numeric_cols, categorical_cols, text_col):
        feature_names = list(numeric_cols)

        if 'cat' in preprocessor.named_transformers_:
            onehot = preprocessor.named_transformers_['cat'].named_steps['onehot']
            feature_names.extend(onehot.get_feature_names_out(categorical_cols))

        if 'text' in preprocessor.named_transformers_:
            tfidf = preprocessor.named_transformers_['text'].named_steps['tfidf']
            tfidf_names = [f"{text_col}_{name}" for name in tfidf.get_feature_names_out()]
            feature_names.extend(tfidf_names)

        return feature_names

    feature_names_transformed = get_feature_names_custom(
        best_rf_pipeline.named_steps["preprocess"],
        numeric_cols,
        categorical_cols,
        text_cols[0]
    )

    # Sample for SHAP explainer
    max_sample_size = X_test_transformed.shape[0]
    sample_size = st.slider(
        "Select number of clients to sample for explanation",
        min_value=1,
        max_value=max_sample_size,
        value=min(500, max_sample_size),
        step=1
    )

    # Random fixed sampling with seed
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(max_sample_size, size=sample_size, replace=False)


    X_sample_transformed = X_test_transformed[sample_indices]
    X_sample_dense = X_sample_transformed.toarray() if hasattr(X_sample_transformed, "toarray") else X_sample_transformed
    X_sample_original = X_test.iloc[sample_indices].reset_index(drop=True)
    y_sample_original = y_test.iloc[sample_indices].reset_index(drop=True)
    y_sample_numeric = y_sample_original.map({'No': 0, 'Yes': 1})

    X_sample_original["customerID"] = X_sample_original["customerID"].astype(str).str.strip()


    # Create SHAP explainer
    @st.cache_data(show_spinner=False)
    def get_explainer(_model, _sample_dense, _feature_names, sample_size):
        return shap.Explainer(_model, _sample_dense, feature_names=_feature_names, approximate=True,)



    explainer = get_explainer(
        best_rf_pipeline.named_steps["classifier"],
        X_sample_dense,
        feature_names_transformed,
        sample_size
    )

    # Compute SHAP values
    @st.cache_data(show_spinner=False)
    def compute_shap_values(_explainer, _sample_dense, sample_size):
        return _explainer(_sample_dense, check_additivity=False)

    shap_values_sample = compute_shap_values(explainer, X_sample_dense,sample_size)

    # Extract SHAP values for positive class
    shap_values_positive_class = shap_values_sample[:, :, 1]

    # Replace scaled numeric values with original values
    def replace_scaled_numeric(shap_values, original_numeric, numeric_cols):
        for i in range(len(numeric_cols)):
            shap_values.data[:, i] = original_numeric[:, i]
        return shap_values

    original_numeric = X_sample_original[numeric_cols].to_numpy()
    shap_values_positive_class = replace_scaled_numeric(shap_values_positive_class, original_numeric, numeric_cols)

    # Ensure customerID is present
    if "customerID" not in X_test.columns:
        st.error("The 'customerID' column is missing from the dataset.")
        st.stop()

    # Build dropdown from sampled clients only
    sampled_client_ids = X_sample_original["customerID"].tolist()
    selected_client_id = st.selectbox("Select a client to explain", sampled_client_ids)

    # Match row from original sample
    matching_row = X_sample_original[X_sample_original["customerID"] == selected_client_id]
    if matching_row.empty:
        st.error(f"No data found for customerID: {selected_client_id}")
        st.stop()

    row_index_sample = matching_row.index[0]
    proba_selected = y_proba[sample_indices[row_index_sample]]
    label_selected = y_pred_custom[sample_indices[row_index_sample]]
    st.subheader(f"Prediction Explanation for Client: {selected_client_id}")
    if label_selected == 1:
        churn_text = f"is predicted to **churn** with a probability of {proba_selected:.2%} (threshold: {chosen_threshold:.2f})"
        action_text = "what actions might help retain similar clients"
    else:
        churn_text = f"is predicted to **stay** with a probability of {(1 - proba_selected):.2%} (threshold: {chosen_threshold:.2f})"
        action_text = "what factors contribute to client retention"

    st.markdown(f"""
    This chart explains **why this specific client {churn_text}**.  
    Each bar shows how much a factor influenced the prediction:
    - **Red bars** push the prediction toward churn.
    - **Blue bars** pull it away from churn.

    This helps you understand the key reasons behind the prediction and {action_text}.
    """)
    try:
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values_positive_class[row_index_sample], show=False)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Could not render waterfall plot: {e}")

    # Global feature importance
    st.subheader("Global Feature Importance")
    st.markdown("""
    This chart shows the **most important factors driving churn across all sampled clients**.  
    The longer the bar, the more influence that feature has on the model’s decision.

    Use this to identify which areas to focus on — like pricing, service quality, or onboarding — to reduce churn overall.
    """)
    
    try:
        fig, ax = plt.subplots()
        shap.summary_plot(
            shap_values_positive_class,
            X_sample_dense,
            feature_names=feature_names_transformed,
            max_display=15,
            show=False
        )
        st.pyplot(fig)
        st.success("""
        Tenure is the most influential factor in predicting churn, with clients who have longer engagement and higher fidelity being less likely to leave. 
                   
        Feedback text also contributes meaningfully to interpretability, while satisfied sentiment generally correlates with retention, others such as the presence of the word "email" are associated with increased churn likelihood.
        """)


    except Exception as e:
        st.error(f"Could not render summary plot: {e}")

    # Filter out text features
    non_text_indices = [i for i, name in enumerate(feature_names_transformed) if not name.startswith(f"{text_cols[0]}_")]
    filtered_shap_values = shap_values_positive_class[:, non_text_indices]
    filtered_feature_names = [feature_names_transformed[i] for i in non_text_indices]

    # Global feature importance (text excluded)
    st.subheader("Global Feature Importance (Text Excluded)")
    st.markdown("""
    This version excludes text-based features to focus on structured data like tenure, charges, and service type.  
    It helps you see which **non-textual factors** are most influential in churn predictions.
    """)

    try:
        fig, ax = plt.subplots()
        shap.summary_plot(
            filtered_shap_values,
            X_sample_dense[:, non_text_indices],
            feature_names=filtered_feature_names,
            max_display=15,
            show=False
        )
        st.pyplot(fig)
        st.success("""
        Tenure remains the most influential factor in predicting churn, reaffirming its central role in client retention.

        Both, the presence of a partner and having children tend to reduce churn likelihood, suggesting that certain personal circumstances may contribute to client stability.

        Gender shows a directional effect:
        - Male clients are more likely to churn
        - Female clients are less likely to churn

        Other structured variables also contribute meaningfully to churn predictions, including:
        - Region
        - Internet service type
        - Payment method
        """)

    except Exception as e:
        st.error(f"Could not render summary plot: {e}")


    st.subheader("SHAP Impact by Numeric Feature Group")
    st.markdown("""
    The following tables present the median SHAP values for three numeric features — **tenure**, **MonthlyCharges**, and **TotalCharges** — segmented by value ranges.

    These features were selected based on their consistent presence in the model and their interpretability in a business context.  
    The tables allow us to observe how the model’s attribution of churn risk varies across different customer profiles defined by these numeric values.
    """)

    def dependence_plot_by_numeric_group(feature_name, shap_values, original_df,original_target, feature_names_transformed):
        if feature_name not in feature_names_transformed:
            st.warning(f"{feature_name} not found in transformed features.")
            return

        idx = feature_names_transformed.index(feature_name)
        shap_column = shap_values.values[:, idx]
        feature_column = original_df[feature_name]

        fig, ax = plt.subplots()

        # Create scatter plot manually using original feature values
        scatter = ax.scatter(
            feature_column,
            shap_column,
            c=original_target, # Color by Churn value
            cmap='plasma',
            alpha=0.4,
            s=10,
            edgecolors='k'
        )


        ax.set_xlabel(feature_name)
        ax.set_ylabel("SHAP value : "+feature_name)
        ax.set_title(f"SHAP Dependence Plot: {feature_name}")
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Churn value")


        st.pyplot(fig)
    for feature in ["tenure", "MonthlyCharges", "TotalCharges"]:
        dependence_plot_by_numeric_group(
            feature,
            shap_values_positive_class,
            X_sample_original,
            y_sample_numeric,
            feature_names_transformed
        )


    st.markdown("""
    The following charts examine how churn is distributed across **TotalCharges** bins for clients with tenure under 24 months.

    This complements the SHAP analysis by showing whether churners in this early stage are concentrated in specific billing ranges.  
    It helps assess whether **mid-range or high TotalCharges** are disproportionately associated with churn among newer clients.
    """)

    def scatter_plot_shap_vs_total_charge(features_df, target_series, shap_values, feature_names_transformed, tenure_threshold=24):
        if "TotalCharges" not in feature_names_transformed:
            st.warning("TotalCharges not found in transformed features.")
            return

        idx = feature_names_transformed.index("TotalCharges")
        shap_total_charge = shap_values.values[:, idx]

        # Filter by tenure and drop NaNs
        mask = (features_df["tenure"] < tenure_threshold) & ~features_df["TotalCharges"].isna() & ~pd.isna(shap_total_charge)
        filtered_df = features_df[mask].copy()
        filtered_df["SHAP_TotalCharges"] = shap_total_charge[mask]
        filtered_df["Churn"] = target_series[mask]

        # Plot
        fig, ax = plt.subplots()
        scatter = ax.scatter(
            filtered_df["TotalCharges"],
            filtered_df["SHAP_TotalCharges"],
            c=filtered_df["Churn"],
            cmap="plasma",
            alpha=0.5,
            s=20,
            edgecolors="k"
        )

        ax.axvline(x=320, color="gray", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_xlabel("TotalCharges")
        ax.set_ylabel("SHAP Value (TotalCharges)")
        ax.set_title("SHAP Attribution vs TotalCharges (Tenure < 24 months)")
        cbar = fig.colorbar(scatter)
        cbar.set_label("Churn (0 = No, 1 = Yes)")
        st.pyplot(fig)


    def summarize_churn_and_shap_by_total_charge(features_df, target_series, shap_values, feature_names_transformed, bin_edges, tenure_threshold=24):
        if "TotalCharges" not in feature_names_transformed:
            st.warning("TotalCharges not found in transformed features.")
            return
        idx = feature_names_transformed.index("TotalCharges")
        shap_total_charge = shap_values.values[:, idx]

        # Filter by tenure
        mask = features_df["tenure"] < tenure_threshold
        filtered_features = features_df[mask].copy()
        filtered_target = target_series[mask]
        filtered_shap = shap_total_charge[mask]

        # Bin TotalCharges
        bin_labels = [f"{int(bin_edges[i]):,}–{int(bin_edges[i+1]):,}" for i in range(len(bin_edges)-1)]
        filtered_features["TotalChargeBin"] = pd.cut(
            filtered_features["TotalCharges"],
            bins=bin_edges,
            labels=bin_labels,
            include_lowest=True
        )

        # Build summary
        summary_df = pd.DataFrame({
            "TotalChargeBin": filtered_features["TotalChargeBin"],
            "Churn": filtered_target,
            "SHAP_TotalCharges": filtered_shap
        })

        grouped = summary_df.groupby("TotalChargeBin").agg(
            count=("Churn", "size"),
            churn_rate=("Churn", "mean"),
            median_shap=("SHAP_TotalCharges", "median")
        )

        st.markdown("#### Churn Rate and Median SHAP Value by TotalCharges Range (Tenure < 24 months)")
        st.dataframe(grouped.style.format({"churn_rate": "{:.1%}", "median_shap": "{:.4f}"}))

    scatter_plot_shap_vs_total_charge(
        features_df=X_sample_original,
        target_series=y_sample_numeric,
        shap_values=shap_values_positive_class,
        feature_names_transformed=feature_names_transformed,
        tenure_threshold=24
    )

    # Define bins manually based on visual inspection
    manual_bin_edges = [0, 320, X_sample_original["TotalCharges"].max()]

    summarize_churn_and_shap_by_total_charge(
        features_df=X_sample_original,
        target_series=y_sample_numeric,
        shap_values=shap_values_positive_class,
        feature_names_transformed=feature_names_transformed,
        bin_edges=manual_bin_edges,
        tenure_threshold=24
    )
    st.success("""
        Clients are most likely to churn within the first 24 months of tenure, indicating a critical window for retention efforts.

        For the general population, SHAP analysis shows that both **very low and very high MonthlyCharges** tend to have a strong negative impact on churn prediction. This suggests that some clients may be more sensitive to pricing extremes, with high-paying clients occasionally showing elevated churn risk.

        Regarding **TotalCharges**, the overall SHAP impact is relatively neutral across most of the population, except for very low values which tend to reduce churn risk.

        However, when isolating clients with **tenure under 24 months**, a clearer segmentation emerges:
        - Clients with **TotalCharges below 320** show a lower churn rate (22.9%) and slightly negative SHAP attribution (–0.0027), indicating stable retention and minimal model concern.
        - Clients with **TotalCharges above 320** exhibit a significantly higher churn rate (35.0%) and a shift to positive SHAP attribution (+0.0003), suggesting that billing accumulation becomes a meaningful churn signal in early tenure.

        This pattern reinforces the importance of monitoring billing exposure during the first months of service, especially as clients cross the low-charge threshold.
        """)





    # Median SHAP for one-hot encoded categories

    st.subheader("SHAP Impact by Categorical Feature")
    st.markdown("""
    The following tables show the median SHAP values for categories within **Region** and **InternetService**.

    These features were selected because several of their encoded categories appeared prominently in the global feature importance analysis.  
    This breakdown provides a clearer view of how specific category memberships influence the model’s churn predictions.
    """)

    def stripplot_shap_for_onehot_category(base_name, shap_values, feature_names, X_dense, original_target):
        st.markdown(f"#### SHAP Stripplot for {base_name} Categories")

        category_indices = [i for i, name in enumerate(feature_names) if name.startswith(base_name + "_")]
        if not category_indices:
            st.info(f"No active categories found for {base_name}")
            return

        plot_data = []

        for i in category_indices:
            feature_name = feature_names[i]
            shap_vals = shap_values.values[:, i]
            active_mask = X_dense[:, i] == 1
            target_vals = original_target[active_mask]

            plot_data.extend([
                {
                    "Category": feature_name.replace(base_name + "_", ""),
                    "SHAP Value": shap_val,
                    "Target": target
                }
                for shap_val, target in zip(shap_vals[active_mask], target_vals)
            ])

        if plot_data:
            df_plot = pd.DataFrame(plot_data)

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.stripplot(
                data=df_plot,
                x="Category",
                y="SHAP Value",
                hue="Target",
                jitter=True,
                alpha=0.4,
                dodge=True,
                ax=ax
            )
            ax.set_title(f"SHAP Stripplot for {base_name} Categories")
            ax.axhline(0, color='gray', linestyle='--', linewidth=1)
            ax.set_ylabel("SHAP Value")
            ax.set_xlabel("Category")
            ax.legend(title="Churn")

            st.pyplot(fig)
        else:
            st.info(f"No SHAP values available for {base_name} categories.")

    stripplot_shap_for_onehot_category(
        "Region",
        shap_values_positive_class,
        feature_names_transformed,
        X_sample_dense,
        y_sample_numeric
    )

    stripplot_shap_for_onehot_category(
        "InternetService",
        shap_values_positive_class,
        feature_names_transformed,
        X_sample_dense,
        y_sample_numeric
    )


    st.success(
        "No region shows a strong standalone influence on churn when accounting for other features in the dataset.  \n"
        "However some regional patterns emerge, for example clients in Region_North tend to exhibit higher retention.\n\n"
        "In contrast, InternetService categories show more distinct effects. Among clients who churn, having fiber optic service often appears as a contributing factor in the model’s prediction, whereas DSL tends to have minimal influence."
    )

    st.subheader("Focused Analysis: Feedback-Based Features")
    st.markdown("""
    The following section concentrates **FeedbackText** — which showed notable influence in the model’s global feature importance analysis.

    These features relate directly to customer interaction and sentiment, and their prominence suggests they play a significant role in predicting churn.  
    """)

    # Filter FeedbackText features from transformed feature names
    feedback_prefix = f"{text_cols[0]}_"
    feedback_indices = [i for i, name in enumerate(feature_names_transformed) if name.startswith(feedback_prefix)]
    feedback_names = [feature_names_transformed[i] for i in feedback_indices]
    feedback_shap_values = shap_values_positive_class.values[:, feedback_indices]
    feedback_values = X_sample_transformed[:, feedback_indices]
    active_mask = feedback_values > 0

    # Compute global mean absolute SHAP values (for ranking)
    mean_abs_shap_full = np.abs(feedback_shap_values).mean(axis=0)

    # Compute directional SHAP values only for active samples (for coloring)
    mean_shap_active = []
    for i in range(feedback_shap_values.shape[1]):
        mask = active_mask[:, i]
        mask_dense = mask.toarray().ravel().astype(bool)
        if np.any(mask_dense):
            values = feedback_shap_values[mask_dense, i]
            mean_shap_active.append(values.mean())
        else:
            mean_shap_active.append(0)
    mean_shap_active = np.array(mean_shap_active)

    # Create dictionaries for importance and direction
    word_importance = {name.replace(feedback_prefix, ""): score for name, score in zip(feedback_names, mean_abs_shap_full)}
    word_direction = {name.replace(feedback_prefix, ""): score for name, score in zip(feedback_names, mean_shap_active)}

    # Split words by direction
    positive_words = {word: word_importance[word] for word, direction in word_direction.items() if direction > 0}
    negative_words = {word: word_importance[word] for word, direction in word_direction.items() if direction < 0}

    # Select top 100 from each
    top_positive = dict(sorted(positive_words.items(), key=lambda x: x[1], reverse=True)[:100])
    top_negative = dict(sorted(negative_words.items(), key=lambda x: x[1], reverse=True)[:100])

    # Normalize for color scaling
    max_val_pos = max(abs(word_direction[word]) for word in top_positive)
    max_val_neg = max(abs(word_direction[word]) for word in top_negative)

    # Define color functions
    def color_func_positive(word, font_size, position, orientation, random_state=None, **kwargs):
        val = word_direction.get(word, 0)
        norm_val = val / max_val_pos if max_val_pos != 0 else 0
        return f"rgb({int(255 * norm_val)}, 0, 0)"  # Red for churn-increasing

    def color_func_negative(word, font_size, position, orientation, random_state=None, **kwargs):
        val = word_direction.get(word, 0)
        norm_val = abs(val) / max_val_neg if max_val_neg != 0 else 0
        return f"rgb(0, {int(255 * norm_val)}, 0)"  # Green for churn-reducing

    # Generate word clouds
    wordcloud_pos = WordCloud(width=800, height=400, background_color="white", color_func=color_func_positive)\
        .generate_from_frequencies(top_positive)

    wordcloud_neg = WordCloud(width=800, height=400, background_color="white", color_func=color_func_negative)\
        .generate_from_frequencies(top_negative)

    # Display in Streamlit
    st.markdown("#### Word Cloud: Feedback Terms Increasing Churn Risk")
    st.markdown("""
    These terms are associated with higher churn likelihood.  
    - **Larger words** indicate stronger overall influence.  
    - **Deeper red** indicates stronger churn-increasing impact.
    """)
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.imshow(wordcloud_pos, interpolation="bilinear")
    ax1.axis("off")
    st.pyplot(fig1)

    st.markdown("#### Word Cloud: Feedback Terms Supporting Retention")
    st.markdown("""
    These terms are associated with lower churn likelihood.  
    - **Larger words** indicate stronger overall influence.  
    - **Deeper green** indicates stronger retention-supporting impact.
    """)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.imshow(wordcloud_neg, interpolation="bilinear")
    ax2.axis("off")
    st.pyplot(fig2)


    # Examples for churn-increasing words
    keywords = ["evening", "emails", "provider","promised"]
    st.markdown("**Examples of churn-increasing feedback terms:**")

    for keyword in keywords:
        st.markdown(f"**Keyword: `{keyword}`**")

        # Add interpretation
        if keyword == "evening":
            st.markdown("_Mentions of 'evening' often relate to performance issues during peak hours, such as slow or unstable internet. These time-specific frustrations may contribute to dissatisfaction and churn._")
        elif keyword == "emails":
            st.markdown("_References to 'emails' typically reflect communication problems, such as ignored support requests. Which can erode trust and increase churn likelihood._")
        elif keyword == "provider":
            st.markdown("_The term 'provider' often appears in negative comparisons or expressions of regret. Suggesting dissatisfaction with service quality or expectations not being met._")
        elif keyword == "promised":
            st.markdown("_Mentions of 'promised' often signal a gap between customer expectations and actual service delivery — especially around speed, pricing, or contract terms. Which can lead to frustration and churn._")
        # Regex pattern to highlight the keyword
        pattern = re.compile(rf"\b({keyword})\b", flags=re.IGNORECASE)
        
        # Filter and highlight examples
        matches = X_sample_original[
            X_sample_original[text_cols[0]].str.contains(keyword, case=False, na=False)
        ][[text_cols[0], "customerID"]].head(2)
        
        for _, row in matches.iterrows():
            highlighted = pattern.sub(r'**\1**', row[text_cols[0]])
            st.markdown(f"- `{row['customerID']}`: _{highlighted}_")

    # Analysis of retention-supporting keyword: "overall"
    st.markdown("### Examples of retention-supporting feedback term:")

    st.markdown("_Mentions of 'overall' often reflect a summary judgment of the customer experience. These comments typically indicate general satisfaction, but may also include constructive suggestions or mild critiques that help identify areas for improvement._")

    # Highlight "overall" in feedback
    keyword = "overall"
    pattern = re.compile(rf"\b({keyword})\b", flags=re.IGNORECASE)
    matches = X_sample_original[
        X_sample_original[text_cols[0]].str.contains(keyword, case=False, na=False)
    ][[text_cols[0], "customerID"]].head(3)

    for _, row in matches.iterrows():
        highlighted = pattern.sub(r'**\1**', row[text_cols[0]])
        st.markdown(f"- `{row['customerID']}`: _{highlighted}_")

    st.markdown("### SHAP Feature Importance for Users Without Feedback")

         # Clients with no feedback data
    total_churn_count = y_sample_numeric.sum()
    missing_text_mask = X_sample_original[text_cols[0]].isna() | (X_sample_original[text_cols[0]].str.strip() == "")
    missing_text_count = missing_text_mask.sum()
    missing_text_churn_count = y_sample_numeric[missing_text_mask].sum()
    churn_share_among_missing = (missing_text_churn_count / total_churn_count) * 100
    st.markdown(f"**Clients with no feedback data:** {missing_text_count} ({(missing_text_count/sample_size)*100:.1f}%)")
    st.markdown(f"**Churners among them:** {missing_text_churn_count}  \n"
                f"- {((missing_text_churn_count/missing_text_count)*100):.1f}% of users without feedback  \n"
                f"- {((missing_text_churn_count/sample_size)*100):.1f}% of total sample  \n"
                f"- {churn_share_among_missing:.1f}% of all churners")

    

    shap_values_non_feedback = shap_values_positive_class[missing_text_mask.values]
    X_non_feedback_dense = X_sample_dense[missing_text_mask.values]

    fig, ax = plt.subplots()
    shap.summary_plot(
        shap_values_non_feedback,
        X_non_feedback_dense,
        feature_names=feature_names_transformed,
        max_display=15,
        show=False
    )
    st.pyplot(fig)


    st.success(
        "The feedback analysis highlights how specific terms relate to churn risk.  \n"
        "Negative comments often mention service interruptions, billing issues, or poor communication — all linked to higher churn likelihood.  \n"
        "Positive comments tend to reflect satisfaction, with terms like 'overall', 'works', and 'good' associated with lower churn predictions.  \n\n"
        "120 users (11.0%) gave no feedback. Among them, 19 churned, 15.8% of that group.  \n"
        "This silent segment includes churners whose reasons for leaving are harder to interpret. Their absence of feedback limits visibility into sentiment."
    )

