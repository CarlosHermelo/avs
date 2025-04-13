# Resumen de Cambios para el Registro de Tokens

## Cambios Realizados

### 1. Función `log_token_summary` en grafo_AGENTE_SERV_flask.py

Se implementó una nueva función que registra de manera clara y estructurada el conteo de tokens tras cada inferencia:

```python
def log_token_summary(tokens_entrada, tokens_salida, modelo):
    """
    Registra un resumen claro del conteo de tokens para cada inferencia.
    """
    # Implementación detallada que genera un bloque formateado en el log
    # y calcula costos aproximados basados en el tipo de modelo
```

Esta función se llama al final del nodo `generate` para registrar:
- Tokens de entrada (pregunta + contexto)
- Tokens de salida (respuesta)
- Total de tokens consumidos
- Costo aproximado

### 2. Integración en el Flujo de Ejecución

Se modificó la función `generate` para llamar a `log_token_summary` justo después de obtener la respuesta del modelo:

```python
def generate(state: MessagesState):
    # Código existente...
    
    # Realizamos la inferencia
    response = llm.invoke(prompt)
    
    # Contamos tokens y registramos
    tokens_salida = contar_tokens(response.content, model_name)
    log_token_summary(tokens_entrada, tokens_salida, model_name)
    
    # Continuación del código existente...
```

### 3. Scripts de Demostración

Se crearon scripts para demostrar y probar el nuevo formato de registro:

1. **standalone_token_counter.py**: 
   - Script independiente que muestra el formato de registro de tokens
   - Útil para pruebas sin depender del sistema completo

2. **README_conteo_tokens.md**:
   - Documentación explicativa sobre el sistema de conteo
   - Instrucciones para analizar los datos del log

## Formato del Registro

El nuevo formato de registro es más claro y estructurado:

```
================================================================================
RESUMEN DE CONTEO DE TOKENS
Fecha y hora: 2025-04-10 23:29:54
Modelo: gpt-4o-mini
================================================================================
TOKENS DE ENTRADA (pregunta + contexto): 110
TOKENS DE SALIDA (respuesta final): 220
TOTAL TOKENS CONSUMIDOS: 330
COSTO APROXIMADO USD: $0.0165
================================================================================
RESUMEN_JSON: {"timestamp": "2025-04-10T23:29:54", "model": "gpt-4o-mini", "input_tokens": 110, "output_tokens": 220, "total_tokens": 330, "approx_cost_usd": 0.0165}
================================================================================
```

## Beneficios

1. **Claridad**: El formato de registro es más legible y estructurado
2. **Información completa**: Incluye todos los datos relevantes en un solo bloque
3. **Análisis facilitado**: El formato JSON permite procesar automáticamente los datos
4. **Estimación de costos**: Incluye cálculos aproximados de costos según el modelo utilizado

## Requisitos de Instalación

El único requisito adicional es la biblioteca `tiktoken`, que ya estaba incluida en el archivo requirements.txt. 