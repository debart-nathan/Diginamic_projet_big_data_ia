
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
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras import layers



from utils.imputation import CustomImputer, TextCleaner
from utils.classify_columns import classify_columns

from sklearn.base import BaseEstimator, TransformerMixin




class AutoencoderWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, pipeline, fit_on_normal_only=True):
        self.pipeline = pipeline
        self.autoencoder = None
        self.fit_on_normal_only = fit_on_normal_only

    def build_autoencoder(self, input_dim):
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
        return autoencoder

    def fit(self, X, y=None):
        # Always fit the pipeline on the full data
        X_scaled_full = self.pipeline.fit_transform(X)
        if hasattr(X_scaled_full, "toarray"):
            X_scaled_full = X_scaled_full.toarray()

        # Filter normal data if requested
        if self.fit_on_normal_only and y is not None:
            y = pd.Series(y, index=X.index)
            normal_mask = y == 0
            if normal_mask.sum() == 0:
                raise ValueError("No normal samples found for training the autoencoder.")
            X_scaled = X_scaled_full[normal_mask.values]
        else:
            X_scaled = X_scaled_full

        input_dim = X_scaled.shape[1]
        self.autoencoder = self.build_autoencoder(input_dim)

        early_stop = keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=5,
            restore_best_weights=True
        )

        self.autoencoder.fit(
            X_scaled,
            X_scaled,
            epochs=50,
            batch_size=32,
            shuffle=True,
            verbose=0,
            callbacks=[early_stop]
        )
        return self

    def transform(self, X):
        X_scaled = self.pipeline.transform(X)
        if hasattr(X_scaled, "toarray"):
            X_scaled = X_scaled.toarray()
        reconstructions = self.autoencoder.predict(X_scaled)
        mse = np.mean(np.square(X_scaled - reconstructions), axis=1)
        return mse
    
    def get_pca_projection(self, X, n_components=2):
        """
        Returns PCA projection of the transformed input data.
        """
        X_scaled = self.pipeline.transform(X)
        if hasattr(X_scaled, "toarray"):
            X_scaled = X_scaled.toarray()

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        return X_pca


def show(df_merged):
    st.header("Customer Churn Detection : Autoencoder Model")

    st.markdown("""
    ### Define Your Business Priorities  
    Use the sliders below to guide the model.
    """)

    @st.cache_resource
    def train_autoencoder(X_train,y_train, _pipeline):
        model = AutoencoderWrapper(pipeline=_pipeline)
        model.fit(X_train,y_train)
        return model


    target = "Churn"
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

    X = df_merged.drop(target, axis=1)
    y = df_merged[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    y_train_numeric = y_train.map({'No': 0, 'Yes': 1})
    # Train and cache the wrapped model
    wrapped_model = train_autoencoder(X_train,y_train_numeric, full_pipeline)
    # Compute reconstruction error
    mse = wrapped_model.transform(X_test)
    y_true = (y_test == "Yes").astype(int)

    # Define slider range
    min_thresh = float(np.min(mse))
    max_thresh = float(np.max(mse))
    default_thresh = float(np.percentile(mse, 95))

    # Add interactive slider
    chosen_threshold = st.slider(
        "Select Reconstruction Error Threshold",
        min_value=min_thresh,
        max_value=max_thresh,
        value=default_thresh,
        step=(max_thresh - min_thresh) / 100
    )

    # Predict anomalies
    y_pred_custom = (mse > chosen_threshold).astype(int)

    # Compute metrics
    true_positives = np.sum((y_pred_custom == 1) & (y_true == 1))
    total_churners = np.sum(y_true)
    predicted_churners = np.sum(y_pred_custom)

    precision_val = true_positives / predicted_churners if predicted_churners > 0 else 0
    recall_val = true_positives / total_churners if total_churners > 0 else 0

    # Display metrics
    st.subheader("Model Performance Based on Your Priorities")
    st.write(f"**Threshold used:** `{chosen_threshold:.5f}`")
    st.metric("Churners Detected", f"{true_positives} of {total_churners}")
    st.metric("Customers Alerted", f"{predicted_churners}")
    st.metric("Coverage (Recall)", f"{recall_val * 100:.1f}%")
    st.metric("Accuracy (Precision)", f"{precision_val * 100:.1f}%")

    st.success(f"""
    **Model Summary Based on Your Priorities**  
    - Churner detection rate: **{recall_val * 100:.1f}%**  
    - Alert precision: **{precision_val * 100:.1f}%**
    """)

    # Classification report
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

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_custom)
    cm_df = pd.DataFrame(cm, index=["No", "Yes"], columns=["No", "Yes"])

    with st.expander("Confusion Matrix"):
        fig, ax = plt.subplots()
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

    # Loss vs. Recall Curve
    thresholds = np.linspace(min_thresh, max_thresh, 100)
    recall_curve = [
        np.sum((mse > t) & (y_true == 1)) / total_churners if total_churners > 0 else 0
        for t in thresholds
    ]

    st.markdown("### Loss vs. Recall Curve")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(thresholds, recall_curve, color="purple", label="Recall")
    ax.axvline(x=chosen_threshold, color="red", linestyle="--", label=f"Threshold @ {chosen_threshold:.5f}")
    ax.set_xlabel("Reconstruction Loss Threshold")
    ax.set_ylabel("Recall")
    ax.set_title("Loss vs. Recall Curve")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # PCA visualization
    X_test_pca = wrapped_model.get_pca_projection(X_test)

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        X_test_pca[:, 0],
        X_test_pca[:, 1],
        c=y_pred_custom,
        cmap='bwr',
        alpha=0.5,
        s=10
    )
    ax.set_title("Résultat de la détection d'anomalies (Autoencoder + PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True)
    st.pyplot(fig)

    # PCA projection
    X_test_pca = wrapped_model.get_pca_projection(X_test)

    # Visualization using true labels
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        X_test_pca[:, 0],
        X_test_pca[:, 1],
        c=y_true,
        cmap='bwr',
        alpha=0.5,
        s=10
    )
    ax.set_title("Répartition réelle des churners (Autoencoder + PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True)
    st.pyplot(fig)