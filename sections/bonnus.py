import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras import layers



from utils.imputation import CustomImputer, TextCleaner
from utils.classify_columns import classify_columns

def find_elbow_point(x, y):
    x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))
    y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))

    line_vec = np.array([x_norm[-1] - x_norm[0], y_norm[-1] - y_norm[0]])
    line_vec /= np.linalg.norm(line_vec)

    vecs = np.stack([x_norm - x_norm[0], y_norm - y_norm[0]], axis=1)
    proj = np.dot(vecs, line_vec)
    proj_point = np.outer(proj, line_vec) + np.array([x_norm[0], y_norm[0]])
    dist = np.linalg.norm(vecs - proj_point, axis=1)

    return x[np.argmax(dist)]

def show(df_merged):
    st.header("Customer Churn Detection : Autoencoder Model")

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

    @st.cache_resource
    def train_autoencoder(_X_train_scaled):
        input_dim = _X_train_scaled.shape[1]
        encoding_dim = int(input_dim * 0.25)

        input_layer = keras.Input(shape=(input_dim,))

        # Encoder
        x = layers.Dense(input_dim, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(input_layer)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(int(input_dim * 0.75), activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(int(input_dim * 0.5), activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.Dropout(0.2)(x)
        encoded = layers.Dense(encoding_dim, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)

        # Decoder
        x = layers.Dense(int(input_dim * 0.5), activation='relu', kernel_regularizer=regularizers.l2(1e-4))(encoded)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(int(input_dim * 0.75), activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(input_dim, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
        decoded = layers.Dense(input_dim, activation='sigmoid')(x)

        autoencoder = keras.Model(inputs=input_layer, outputs=decoded)
        autoencoder.compile(optimizer='adam', loss='mse')

        early_stop = keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=5,
            restore_best_weights=True
        )

        autoencoder.fit(
            _X_train_scaled,
            _X_train_scaled,
            epochs=50,
            batch_size=32,
            shuffle=True,
            verbose=0,
            callbacks=[early_stop]
        )

        return autoencoder


    df_aggregated = df_merged.groupby("customerID", as_index=False, observed=False).agg({
        col: "median" if df_merged[col].dtype.kind in "iufc" else "first"
        for col in df_merged.columns
    })

    target = "Churn"
    X = df_aggregated.drop(target, axis=1)
    y = df_aggregated[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)



    numeric_cols, categorical_cols, text_cols = classify_columns(df_merged, target)


    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    text_transformer = Pipeline([
        ("cleaner", TextCleaner()),
        ("tfidf", TfidfVectorizer(
            stop_words='english',
            token_pattern=r'\b[a-zA-Z]{2,}\b'
        ))
    ])

    # Only one text column allowed for TfidfVectorizer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
            ("text", text_transformer, text_cols[0])
        ],
        sparse_threshold=0  # Force dense output
    )

    full_pipeline = Pipeline([
        ("custom_imputer", CustomImputer()),
        ("preprocess", preprocessor)
    ])

    X_train_nonchurn = X_train[y_train == "No"]
    X_train_scaled = full_pipeline.fit_transform(X_train_nonchurn)
    X_test_scaled = full_pipeline.transform(X_test)
    if hasattr(X_train_scaled, "toarray"):
        X_train_scaled = X_train_scaled.toarray()
    if hasattr(X_test_scaled, "toarray"):
        X_test_scaled = X_test_scaled.toarray()
    autoencoder = train_autoencoder(X_train_scaled)
    reconstructions = autoencoder.predict(X_test_scaled)
    mse = np.mean(np.square(X_test_scaled - reconstructions), axis=1)
    y_true = (y_test == "Yes").astype(int)

    thresholds = np.linspace(min(mse), max(mse), 100)
    precision, recall = [], []

    for t in thresholds:
        preds = (mse > t).astype(int)
        p, r, _, _ = precision_recall_fscore_support(y_true, preds, average='binary', zero_division=0)
        precision.append(p)
        recall.append(r)


    elbow_threshold = find_elbow_point(thresholds, recall)

    # Filter thresholds that meet user-defined minimums
    valid_indices = [i for i, (p, r) in enumerate(zip(precision, recall)) if p >= min_precision and r >= min_recall]

   
    elbow_threshold = find_elbow_point(thresholds, recall)


    valid_indices = [
        i for i, (p, r) in enumerate(zip(precision, recall))
        if p >= min_precision and r >= min_recall
    ]
    if valid_indices:
        chosen_idx = min(valid_indices, key=lambda i: abs(thresholds[i] - elbow_threshold))
        chosen_threshold = thresholds[chosen_idx]
    else:
        st.warning("No threshold meets both minimum coverage and accuracy.")

        # Use a default threshold (e.g., 95th percentile of reconstruction loss)
        default_threshold = np.percentile(mse, 95)
        y_pred_custom = (mse > default_threshold).astype(int)

        # Generate classification report
        report_dict = classification_report(
            y_true,
            y_pred_custom,
            target_names=["No", "Yes"],
            zero_division=0,
            output_dict=True
        )
        report_df = pd.DataFrame(report_dict).transpose()

        with st.expander("Default Model Performance (Threshold = 95th Percentile)"):
            st.dataframe(report_df.style.format("{:.2f}"))

        # Optionally show confusion matrix
        cm = confusion_matrix(y_true, y_pred_custom)
        cm_df = pd.DataFrame(cm, index=["No", "Yes"], columns=["No", "Yes"])
        with st.expander("Confusion Matrix"):
            fig, ax = plt.subplots()
            sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted Labels")
            ax.set_ylabel("True Labels")
            ax.set_title("Confusion Matrix (Default Threshold)")
            st.pyplot(fig)

        return



    y_pred_custom = (mse > chosen_threshold).astype(int)

    total_churners = sum(y_true)
    predicted_churners = sum(y_pred_custom)
    true_positives = sum((y_pred_custom == 1) & (y_true == 1))

    try:
        threshold_idx = np.where(np.isclose(thresholds, chosen_threshold))[0][0]
    except IndexError:
        threshold_idx = np.argmin(np.abs(thresholds - chosen_threshold))

    recall_val = recall[threshold_idx]
    precision_val = precision[threshold_idx]

    st.subheader("Model Performance Based on Your Priorities")
    st.write(f"**Threshold used:** `{chosen_threshold:.5f}`")
    st.metric("Churners Detected", f"{true_positives} of {total_churners}")
    st.metric("Customers Alerted", f"{predicted_churners}")
    st.metric("Coverage (Recall)", f"{recall_val * 100:.1f}%")
    st.metric("Accuracy (Precision)", f"{precision_val * 100:.1f}%")


    st.success(f"""
    **Model Summary Based on Your Priorities**  
    - Churner detection rate: **{recall[np.argmax(thresholds == chosen_threshold)]*100:.1f}%**  
    - Alert precision: **{precision[np.argmax(thresholds == chosen_threshold)]*100:.1f}%**  
    """)

    report_dict = classification_report(
        y_true,
        y_pred_custom,
        target_names=["No", "Yes"],
        zero_division=0,
        output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose()

    with st.expander("Detailed Classification Report"):
        st.dataframe(report_df.style.format("{:.2f}"))

    cm = confusion_matrix(y_true, y_pred_custom)
    cm_df = pd.DataFrame(cm, index=["No", "Yes"], columns=["No", "Yes"])

    with st.expander("Confusion Matrix"):
        fig, ax = plt.subplots()
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

    st.markdown("### Detection Tradeoff Curve")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, label="Precision-Recall Curve", color="blue")
    ax.axvline(x=recall[np.argmax(thresholds == chosen_threshold)], color="red", linestyle="--", label=f"Coverage @ {recall[np.argmax(thresholds == chosen_threshold)]:.2f}")
    ax.axhline(y=precision[np.argmax(thresholds == chosen_threshold)], color="green", linestyle="--", label=f"Accuracy @ {precision[np.argmax(thresholds == chosen_threshold)]:.2f}")
    ax.set_xlabel("Coverage (Recall)")
    ax.set_ylabel("Accuracy (Precision)")
    ax.set_title("Detection Tradeoff for Churners")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.markdown("### Loss vs. Recall Curve")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(thresholds, recall, color="purple", label="Recall")
    ax.axvline(x=chosen_threshold, color="red", linestyle="--", label=f"Threshold @ {chosen_threshold:.5f}")
    ax.set_xlabel("Reconstruction Loss Threshold")
    ax.set_ylabel("Recall")
    ax.set_title("Loss vs. Recall Curve")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)