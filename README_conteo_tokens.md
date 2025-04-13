# Conteo de Tokens en el Sistema PAMI

## Resumen

Este documento explica cómo revisar y entender el conteo de tokens que se registra en el archivo de log durante el funcionamiento del sistema. Esta información es crucial para:

1. Monitorear el consumo de recursos de OpenAI
2. Estimar costos del servicio
3. Optimizar las consultas y contextos

## ¿Qué se registra?

Para cada consulta que procesa el sistema, se registra:

- **Tokens de entrada (pregunta + contexto)**: Total de tokens utilizados en la consulta incluyendo:
  - La pregunta original del usuario
  - El contexto recuperado de la base vectorial
  
- **Tokens de salida (respuesta)**: Total de tokens generados en la respuesta

- **Total de tokens consumidos**: Suma de tokens de entrada y salida

- **Costo aproximado**: Estimación del costo en USD basado en los precios actuales de OpenAI

## Dónde encontrar esta información

La información se guarda en el archivo `script_log.log` en la raíz del proyecto.

### Formato del registro en el log

Cada inferencia genera un bloque como el siguiente:

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
RESUMEN_JSON: {"timestamp": "2025-04-10T23:29:54.954925", "model": "gpt-4o-mini", "input_tokens": 110, "output_tokens": 220, "total_tokens": 330, "approx_cost_usd": 0.0165}
================================================================================
```

## Cómo analizar los datos

Para analizar estos datos puedes:

1. Buscar en el archivo `script_log.log` usando herramientas como `grep` o un editor de texto.
   ```
   grep "RESUMEN DE CONTEO DE TOKENS" script_log.log -A 10
   ```

2. Extraer los datos JSON para análisis automatizado con un script simple como:
   ```python
   import json
   
   token_data = []
   with open('script_log.log', 'r') as f:
       for line in f:
           if 'RESUMEN_JSON:' in line:
               # Extraer el JSON
               json_str = line.split('RESUMEN_JSON:', 1)[1].strip()
               token_data.append(json.loads(json_str))
   
   # Análisis básico
   total_tokens = sum(item['total_tokens'] for item in token_data)
   total_cost = sum(item['approx_cost_usd'] for item in token_data)
   
   print(f"Total consultas: {len(token_data)}")
   print(f"Total tokens consumidos: {total_tokens}")
   print(f"Costo total estimado: ${total_cost:.4f}")
   ```

## Ejemplo de uso

El script `standalone_token_counter.py` muestra cómo funciona el conteo de tokens y genera una salida de ejemplo en `token_test_log.log`. Puedes ejecutarlo para ver una demostración:

```
python standalone_token_counter.py
```

## Notas sobre el precio de los tokens

Los precios actuales utilizados para el cálculo son:

- **Modelos GPT-4**:
  - Tokens de entrada: $0.03 por 1K tokens
  - Tokens de salida: $0.06 por 1K tokens

- **Modelos GPT-3.5**:
  - $0.002 por 1K tokens (tanto entrada como salida)

Estos precios pueden variar según las tarifas actuales de OpenAI. 