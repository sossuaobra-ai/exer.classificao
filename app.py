import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ---------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA E ESTILO
# ---------------------------------------------------------
st.set_page_config(
    page_title="Controle de Qualidade - Usinagem",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Controle de Qualidade de Peças Usinadas")
st.markdown("""
Esta aplicação utiliza um modelo de **Support Vector Machine (SVM)** para classificar o estado de peças de usinagem
com base no **Peso** e na **Espessura**.
""")

# ---------------------------------------------------------
# GERACÃO DE DADOS SINTÉTICOS E TREINAMENTO DO MODELO
# ---------------------------------------------------------
@st.cache_data
def carregar_dados_e_modelo():
    # Classes: 0: Aprovada, 1: Retrabalho, 2: Refugo
    # Centróides definidos para simular valores de peso (g) e espessura (mm)
    centers = [
        [150.0, 10.0],  # Aprovada (cerca de 150g, 10mm)
        [158.0, 10.8],  # Retrabalho (pevemente acima da tolerância)
        [170.0, 12.5]   # Refugo (muito acima das dimensões)
    ]
    
    X, y = make_blobs(
        n_samples=600,
        centers=centers,
        cluster_std=[2.0, 2.5, 3.0],
        random_state=42
    )
    
    df = pd.DataFrame(X, columns=['Peso_gramas', 'Espessura_mm'])
    
    # Mapeamento numérico -> Rótulo
    mapeamento = {0: 'Aprovada', 1: 'Retrabalho', 2: 'Refugo'}
    df['Status'] = [mapeamento[val] for val in y]
    
    # Divisão em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        df[['Peso_gramas', 'Espessura_mm']], 
        y, 
        test_state=42, 
        test_size=0.25, 
        stratify=y
    )
    
    # Treino do Modelo SVM
    modelo_svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    modelo_svm.fit(X_train, y_train)
    
    # Avaliação no conjunto de teste
    y_pred = modelo_svm.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    return modelo_svm, cm, mapeamento, df

# Carrega modelo e dados (cached para melhor performance)
modelo_svm, cm, mapeamento, df_dados = carregar_dados_e_modelo()

# ---------------------------------------------------------
# LAYOUT PRINCIPAL DA APLICAÇÃO
# ---------------------------------------------------------
col_inspeçao, col_relatorio = st.columns([1, 1], gap="large")

with col_inspeçao:
    st.subheader("📋 Inspecionar Nova Peça")
    st.write("Insira as medições da peça obtidas na bancada de medição:")
    
    # Sliders para entrada dos dados do inspetor
    peso_input = st.slider(
        "Peso da Peça (gramas)", 
        min_value=130.0, 
        max_value=190.0, 
        value=150.0, 
        step=0.1
    )
    
    espessura_input = st.slider(
        "Espessura da Peça (mm)", 
        min_value=5.0, 
        max_value=16.0, 
        value=10.0, 
        step=0.05
    )
    
    btn_avaliar = st.button("🔍 Avaliar Peça", type="primary", use_container_width=True)
    
    if btn_avaliar:
        # Predição
        entrada = np.array([[peso_input, espessura_input]])
        classe_predita = modelo_svm.predict(entrada)[0]
        status = mapeamento[classe_predita]
        
        st.markdown("---")
        st.subheader("Resultado da Inspeção:")
        
        # Exibição do resultado com alertas de cor
        if status == 'Aprovada':
            st.success(f"✅ **Status: APROVADA**\nA peça atende a todos os critérios de tolerância e qualidade.")
        elif status == 'Retrabalho':
            st.warning(f"⚠️ **Status: RETRABALHO**\nA peça está fora de tolerância, mas pode ser ajustada usinando novamente.")
        else: # Refugo
            st.error(f"🚨 **Status: REFUGO**\nA peça está criticamente fora das especificações e deve ser descartada.")

with col_relatorio:
    st.subheader("📊 Desempenho do Modelo (SVM)")
    st.write("Matriz de confusão calculada sobre o conjunto de teste:")
    
    # Plot da Matriz de Confusão com Matplotlib / Seaborn
    labels = ['Aprovada', 'Retrabalho', 'Refugo']
    fig, ax = plt.subplots(figsize=(5, 4))
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=labels, 
        yticklabels=labels,
        ax=ax,
        cbar=False
    )
    
    ax.set_xlabel('Classe Predita')
    ax.set_ylabel('Classe Real')
    ax.set_title('Matriz de Confusão')
    
    st.pyplot(fig)

# ---------------------------------------------------------
# SEÇÃO EXPANSÍVEL: VISUALIZAÇÃO DOS DADOS HISTÓRICOS
# ---------------------------------------------------------
with st.expander("👀 Ver dados de treino sintéticos"):
    st.dataframe(df_dados.head(20), use_container_width=True)
