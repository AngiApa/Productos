import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import joblib

# Configuración de la barra lateral con un menú de opciones
with st.sidebar:
    selected = option_menu(
        menu_title="MyLogis",  # Título del menú
        options=["Inicio", "Acerca de"],  # Opciones del menú
        icons=["house", "info"],  # Iconos para cada opción (basados en Bootstrap)
        menu_icon="cast",  # Icono del menú principal
        default_index=0,  # Índice predeterminado (opción seleccionada por defecto)
    )

# Mostrar contenido según la opción seleccionada
if selected == "Inicio":
    st.title("🔍 Análisis - MyLogis")

    # Estructura del flujo de trabajo
    tabs = st.tabs([
        "📂 Cargar Datos",
        "📊 Seleccionar Variables",
        "✂️ Dividir Datos",
        "🤖 Entrenar Modelo",
        "📈 Evaluar Modelo",
        "🔮 Predicción con Nuevos Datos"
    ])

    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'X' not in st.session_state:
        st.session_state.X = None
    if 'y' not in st.session_state:
        st.session_state.y = None
    if 'X_train_scaled' not in st.session_state:
        st.session_state.X_train_scaled = None
    if 'X_test_scaled' not in st.session_state:
        st.session_state.X_test_scaled = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None

    with tabs[0]:
        st.markdown(
            """
            <style>
            .header {
                font-size: 28px;
                font-weight: bold;
                color: #4CAF50;
                text-align: center;
                margin-bottom: 20px;
            }
            .instructions {
                font-size: 16px;
                color: #555;
                line-height: 1.6;
                margin-bottom: 20px;
            }
            .subheader {
                font-size: 20px;
                font-weight: bold;
                color: #333;
                margin-top: 20px;
            }
            .dataframe {
                margin-top: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            </style>
            <div class="header">📂 Cargar Datos</div>
            """,
            unsafe_allow_html=True,
        )

        # Instrucciones (Opción 7)
        st.markdown(
            """
            <div class="instructions">
            <strong>Instrucciones para cargar datos:</strong>
            <ul>
                <li>Archivos permitidos: CSV o Excel (XLSX).</li>
                <li>Asegúrate de que el archivo no tenga celdas fusionadas (en caso de Excel).</li>
                <li>La primera fila debe contener los nombres de las columnas.</li>
                <li>Si el archivo contiene valores nulos, puedes tratarlos en pasos posteriores.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Subir archivo
        uploaded_file = st.file_uploader(
            "Sube un archivo de datos (CSV o XLSX)", type=["csv", "xlsx"]
        )

        if uploaded_file:
            try:
                # Leer el archivo subido
                if uploaded_file.name.endswith(".csv"):
                    st.session_state.df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(".xlsx"):
                    st.session_state.df = pd.read_excel(uploaded_file)

                # Vista previa de los datos cargados
                st.markdown('<div class="subheader">Vista previa de los datos cargados:</div>', unsafe_allow_html=True)
                st.dataframe(st.session_state.df.head(), use_container_width=True)

                # Análisis básico de los datos (Opción 3)
                st.markdown('<div class="subheader">Análisis Básico de los Datos:</div>', unsafe_allow_html=True)
                st.markdown(f"- **Dimensiones del Dataset:** {st.session_state.df.shape[0]} filas, {st.session_state.df.shape[1]} columnas")
                
                # Tipos de datos
                st.markdown('<div class="subheader">Tipos de Datos:</div>', unsafe_allow_html=True)
                st.dataframe(st.session_state.df.dtypes, use_container_width=True)

                # Resumen estadístico
                st.markdown('<div class="subheader">Resumen Estadístico:</div>', unsafe_allow_html=True)
                st.dataframe(st.session_state.df.describe(), use_container_width=True)

            except Exception as e:
                st.error(f"Error al cargar los datos: {e}")
        else:
            st.info("Por favor, carga un archivo para continuar.")


    # TAB Seleccionar Variables
    with tabs[1]:
        st.header("📊 Seleccionar Variables")

        if "df" not in st.session_state or st.session_state.df is None:
            st.warning("Por favor, carga los datos primero.")
        else:
            columns = st.session_state.df.columns.tolist()

            # Selección de variables
            target_variable = st.selectbox("Selecciona la variable objetivo (Y):", columns)
            feature_variables = st.multiselect(
                "Selecciona las variables predictoras (X):",
                [col for col in columns if col != target_variable]
            )

            if target_variable and feature_variables:
                # Preparar variables X e Y
                st.session_state.X = st.session_state.df[feature_variables]
                st.session_state.y = st.session_state.df[target_variable]
                st.success(f"Variable objetivo: {target_variable}")
                st.info(f"Variables predictoras: {', '.join(feature_variables)}")

                # Opción 1: Visualización de distribución de variables predictoras (X)
                st.write("### Distribución de las Variables Predictoras (X)")
                for feature in feature_variables:
                    st.write(f"#### Distribución de {feature}")
                    if st.session_state.df[feature].dtype in ['int64', 'float64']:
                        fig = px.histogram(
                            st.session_state.df,
                            x=feature,
                            title=f"Distribución de {feature}",
                            nbins=30,
                            labels={"x": feature, "y": "Frecuencia"},
                            marginal="box"  # Agrega un boxplot a un lado
                        )
                    else:
                        fig = px.bar(
                            st.session_state.df[feature].value_counts(),
                            title=f"Distribución de {feature}",
                            labels={"index": feature, "value": "Frecuencia"}
                        )
                    st.plotly_chart(fig)

                # Opción 2: Matriz de Correlación con Plotly
                st.write("### Matriz de Correlación")
                correlation_matrix = st.session_state.df[feature_variables].corr()
                fig_corr = px.imshow(
                    correlation_matrix,
                    text_auto=".2f",
                    color_continuous_scale="RdBu_r",
                    labels=dict(color="Correlación"),
                    title="Matriz de Correlación entre Variables Seleccionadas"
                )
                fig_corr.update_layout(
                    title_font_size=20,
                    title_x=0.5,
                    xaxis_title="Variables",
                    yaxis_title="Variables",
                    coloraxis_showscale=True,
                    width=800,
                    height=600
                )
                st.plotly_chart(fig_corr)

                # Opción 3: Relación entre las Variables Seleccionadas
                st.write("### Relación entre Variables Predictoras Seleccionadas")
                for i, feature_x in enumerate(feature_variables):
                    for feature_y in feature_variables[i + 1:]:
                        st.write(f"#### Relación entre {feature_x} y {feature_y}")
                        if st.session_state.df[feature_x].dtype in ['int64', 'float64'] and st.session_state.df[feature_y].dtype in ['int64', 'float64']:
                            fig = px.scatter(
                                st.session_state.df,
                                x=feature_x,
                                y=feature_y,
                                labels={"x": feature_x, "y": feature_y},
                                title=f"Relación entre {feature_x} y {feature_y}"
                            )
                        else:
                            fig = px.box(
                                st.session_state.df,
                                x=feature_x,
                                y=feature_y,
                                labels={"x": feature_x, "y": feature_y},
                                title=f"Distribución entre {feature_x} y {feature_y}"
                            )
                        st.plotly_chart(fig)

    with tabs[2]:
        st.header("✂️ Dividir Datos")

        if "X" not in st.session_state or "y" not in st.session_state or st.session_state.X is None or st.session_state.y is None:
            st.warning("Por favor, selecciona las variables primero.")
        else:
            # Parámetros de división
            test_size = st.slider("Tamaño del conjunto de prueba:", 0.1, 0.5, 0.3)
            random_state = st.number_input("Semilla de aleatoriedad:", value=42, step=1)

            # División de datos
            st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = train_test_split(
                st.session_state.X, st.session_state.y, test_size=test_size, random_state=random_state
            )

            # Selección del escalador
            scale_data = st.checkbox("¿Escalar los datos?", value=True)
            scaler_option = st.selectbox(
                "Selecciona el método de escalado:",
                ["StandardScaler", "MinMaxScaler", "Ninguno"]
            )

            if scale_data and scaler_option != "Ninguno":
                if scaler_option == "StandardScaler":
                    st.session_state.scaler = StandardScaler()
                elif scaler_option == "MinMaxScaler":
                    st.session_state.scaler = MinMaxScaler()

                st.session_state.X_train_scaled = st.session_state.scaler.fit_transform(st.session_state.X_train)
                st.session_state.X_test_scaled = st.session_state.scaler.transform(st.session_state.X_test)
                st.success(f"Datos escalados utilizando {scaler_option}.")
            else:
                st.session_state.X_train_scaled = st.session_state.X_train
                st.session_state.X_test_scaled = st.session_state.X_test

            # Tamaños de los conjuntos
            train_size = st.session_state.X_train.shape[0]
            test_size = st.session_state.X_test.shape[0]
            st.write("### Tamaños de los conjuntos:")
            st.write(f"- **Conjunto de entrenamiento:** {train_size} muestras")
            st.write(f"- **Conjunto de prueba:** {test_size} muestras")

            # Visualización del tamaño de los conjuntos
            fig_sizes = px.bar(
                x=["Entrenamiento", "Prueba"],
                y=[train_size, test_size],
                labels={"x": "Conjuntos", "y": "Tamaño"},
                title="Tamaño de los Conjuntos"
            )
            st.plotly_chart(fig_sizes)

            # Comparación de distribuciones antes y después del escalado
            st.write("### Comparación de Distribuciones (Antes y Después del Escalado)")
            feature_to_plot = st.selectbox("Selecciona una variable para comparar:", st.session_state.X.columns)

            if scale_data and scaler_option != "Ninguno":
                original_data = st.session_state.X_train[feature_to_plot]
                scaled_data = st.session_state.X_train_scaled[:, st.session_state.X.columns.get_loc(feature_to_plot)]

                # Plotly para comparar distribuciones
                fig = px.histogram(
                    x=[original_data, scaled_data],
                    labels={"x": feature_to_plot, "color": "Datos"},
                    title=f"Distribución de {feature_to_plot} (Original vs Escalado)",
                    barmode="overlay",
                    color_discrete_sequence=["blue", "orange"]
                )
                fig.update_layout(
                    legend_title="Tipo de Datos",
                    legend=dict(
                        itemsizing="constant",
                        orientation="h",
                        yanchor="top",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                st.plotly_chart(fig)


    
    # Paso 4: Entrenar Modelo
    with tabs[3]:
        st.header("🤖 Entrenar Modelo")

        if "X_train_scaled" not in st.session_state or st.session_state.X_train_scaled is None:
            st.warning("Por favor, divide los datos primero.")
        else:
            # Selección de modelo: Regresión Logística
            st.write("### Modelo: Regresión Logística")
            C_value = st.number_input("Valor de regularización (C):", value=1.0, step=0.1)
            max_iter = st.number_input("Máximo número de iteraciones:", value=100, step=10)

            # Entrenar el modelo
            st.session_state.model = LogisticRegression(C=C_value, max_iter=max_iter)
            st.session_state.model.fit(st.session_state.X_train_scaled, st.session_state.y_train)
            st.success("Modelo de Regresión Logística entrenado con éxito.")

            # Métricas de evaluación
            st.write("### Evaluación del Modelo")
            y_pred = st.session_state.model.predict(st.session_state.X_test_scaled)

            # Precisión
            accuracy = accuracy_score(st.session_state.y_test, y_pred)
            st.write(f"**Precisión del modelo:** {accuracy:.2f}")


            # Coeficientes del modelo
            st.write("### Coeficientes del Modelo")
            coefficients = st.session_state.model.coef_[0]
            intercept = st.session_state.model.intercept_[0]

            coef_df = pd.DataFrame({
                "Características": st.session_state.X.columns,
                "Coeficientes": coefficients
            })
            coef_df.sort_values(by="Coeficientes", ascending=False, inplace=True)

            # Visualización de coeficientes
            st.write("#### Tabla de Coeficientes")
            fig_coefs = px.bar(
                coef_df,
                x="Coeficientes",
                y="Características",
                orientation="h",
                title="Importancia de los Coeficientes (Regresión Logística)",
                labels={"Coeficientes": "Valor del Coeficiente", "Características": "Características"}
            )
            st.plotly_chart(fig_coefs)

            # Tabla estilizada
            st.write("#### Coeficientes en Tabla")
            coef_df_styled = coef_df.style.background_gradient(cmap="coolwarm", subset=["Coeficientes"])
            st.dataframe(coef_df_styled, use_container_width=True)


    # Paso 5: Evaluar Modelo
    with tabs[4]:
        st.header("📈 Evaluar Modelo")

        if "model" not in st.session_state or st.session_state.model is None:
            st.warning("Por favor, entrena el modelo primero.")
        else:
            # Predicciones
            y_pred = st.session_state.model.predict(st.session_state.X_test_scaled)

            # Métricas de Evaluación
            accuracy = accuracy_score(st.session_state.y_test, y_pred)
            st.success(f"Exactitud del modelo: {accuracy:.2f}")

            # Opción 2: Visualización Mejorada de la Matriz de Confusión
            from sklearn.metrics import confusion_matrix
            import plotly.express as px

            cm = confusion_matrix(st.session_state.y_test, y_pred)
            st.write("### Matriz de Confusión")
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                labels=dict(x="Predicción", y="Real", color="Frecuencia"),
                title="Matriz de Confusión",
                color_continuous_scale="Blues"
            )
            fig_cm.update_layout(
                title_font_size=20,
                title_x=0.5,
                xaxis_title="Predicción",
                yaxis_title="Real"
            )
            st.plotly_chart(fig_cm)

            # Opción 3: Curva ROC y AUC
            from sklearn.metrics import roc_curve, auc
            import matplotlib.pyplot as plt

            if len(st.session_state.y_test.unique()) == 2:  # Solo aplica para clasificación binaria
                y_prob = st.session_state.model.predict_proba(st.session_state.X_test_scaled)[:, 1]
                fpr, tpr, _ = roc_curve(st.session_state.y_test, y_prob)
                roc_auc = auc(fpr, tpr)

                st.write("### Curva ROC y AUC")
                fig_roc, ax_roc = plt.subplots()
                ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                ax_roc.plot([0, 1], [0, 1], "k--", label="AUC Base (0.5)")
                ax_roc.set_title("Curva ROC")
                ax_roc.set_xlabel("Tasa de Falsos Positivos (FPR)")
                ax_roc.set_ylabel("Tasa de Verdaderos Positivos (TPR)")
                ax_roc.legend(loc="lower right")
                st.pyplot(fig_roc)

            # Opción 4: Comparación de Predicciones
            st.write("### Comparación de Predicciones")
            import pandas as pd
            comparison_df = pd.DataFrame({
                "Real": st.session_state.y_test,
                "Predicción": y_pred
            })
            st.dataframe(comparison_df)


    # Paso 6: Predicción con Nuevos Datos
    with tabs[5]:
        st.header("🔮 Predicción con Nuevos Datos")

        if st.session_state.model is None:
            st.warning("Por favor, entrena el modelo primero.")
        else:
            # Entrada manual
            new_data = {}
            for feature in st.session_state.X.columns:
                value = st.number_input(f"Valor para {feature}:", value=0.0)
                new_data[feature] = [value]

            if st.button("Realizar Predicción"):
                new_df = pd.DataFrame(new_data)
                new_scaled = st.session_state.scaler.transform(new_df)
                prediction = st.session_state.model.predict(new_scaled)
                probabilities = st.session_state.model.predict_proba(new_scaled)

                st.success(f"Predicción: {prediction[0]}")
                
                # Opción 1: Mostrar probabilidades asociadas a cada clase
                st.write("### Probabilidades de las Clases")
                prob_df = pd.DataFrame(probabilities, columns=st.session_state.model.classes_)
                st.dataframe(prob_df)

            # Interpretabilidad del modelo
            st.write("### Interpretabilidad del Modelo")
            
            # Opción 4: Mostrar importancia de características
            if hasattr(st.session_state.model, "coef_"):  # Para Regresión Logística
                coefficients = st.session_state.model.coef_[0]
                feature_importance_df = pd.DataFrame({
                    "Características": st.session_state.X.columns,
                    "Coeficientes": coefficients
                }).sort_values(by="Coeficientes", ascending=False)

                st.write("#### Coeficientes del Modelo")
                st.dataframe(feature_importance_df)


                # Gráfico sigmoide
                st.write("### Gráfico Sigmoide")
                import numpy as np
                import matplotlib.pyplot as plt

                def sigmoid(z):
                    return 1 / (1 + np.exp(-z))

                z = np.linspace(-10, 10, 100)
                y = sigmoid(z)

                fig_sigmoid, ax_sigmoid = plt.subplots()
                ax_sigmoid.plot(z, y, label="Sigmoide", color="blue")
                ax_sigmoid.axhline(0.5, color="red", linestyle="--", label="Probabilidad 0.5")
                ax_sigmoid.set_title("Función Sigmoide")
                ax_sigmoid.set_xlabel("Valor (z)")
                ax_sigmoid.set_ylabel("Probabilidad")
                ax_sigmoid.legend()
                st.pyplot(fig_sigmoid)



elif selected == "Acerca de":
    # Estilos CSS para la sección
    st.markdown(
        """
        <style>
        .about-title {
            font-size: 36px;
            font-weight: bold;
            color: #4CAF50;
            text-align: center;
            margin-bottom: 20px;
        }
        .about-section {
            font-size: 18px;
            line-height: 1.6;
            color: #555;
            margin-bottom: 20px;
        }
        .highlight {
            color: #f44336;
            font-weight: bold;
        }
        .code-block {
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            font-family: monospace;
            margin-bottom: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Título
    st.markdown('<div class="about-title">📖 Acerca de la Regresión Logística</div>', unsafe_allow_html=True)

    # Descripción general
    st.markdown(
        """
        <div class="about-section">
        La <span class="highlight">regresión logística</span> es una técnica de aprendizaje supervisado utilizada 
        para resolver problemas de clasificación. A diferencia de la regresión lineal, que predice valores continuos, 
        la regresión logística modela la probabilidad de que un evento ocurra, utilizando la función sigmoide para transformar los valores de salida.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Cómo funciona
    st.markdown('<div class="about-section"><b>¿Cómo funciona?</b></div>', unsafe_allow_html=True)

    st.latex(r"""
    1. \, \text{Calcula una combinación lineal de las características de entrada:} \\
    z = w_0 + w_1x_1 + w_2x_2 + \dots + w_nx_n
    """)

    st.latex(r"""
    2. \, \text{Transforma el resultado mediante la función sigmoide:} \\
    \sigma(z) = \frac{1}{1 + e^{-z}}
    """)

    st.latex(r"""
    3. \, \text{Interpreta la salida como una probabilidad:} \\
    P(Y=1|X) = \sigma(z)
    """)


    # Ventajas y desventajas
    st.markdown(
        """
        <div class="about-section">
        **Ventajas de la Regresión Logística:** <br>
        - Es fácil de interpretar gracias a sus coeficientes. <br>
        - Funciona bien con problemas linealmente separables. <br>
        - Proporciona probabilidades como salida, útiles para decisiones basadas en confianza. <br><br>

        Limitaciones: <br>
        - No captura relaciones no lineales entre características. <br>
        - Sensible a datos desbalanceados. <br>
        - Puede requerir escalado de las características.
        </div>
        """,
        unsafe_allow_html=True,
    )

    



