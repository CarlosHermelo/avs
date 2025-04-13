from grafo_AGENTE_SERV_flask import log_message, contar_tokens, log_token_summary

# Simular un conteo de tokens para verificar el funcionamiento
pregunta = "¿Qué servicios ofrece PAMI para personas mayores?"
contexto = """
PAMI ofrece varios servicios para personas mayores:
- Bolsón alimentario
- Apoyo a la dependencia y fragilidad
- Viviendas en comodato
- Atención médica domiciliaria
- Club de día para personas mayores

Se requiere documentación específica para cada servicio:
- DNI
- Credencial de afiliación
- Informes médicos (según corresponda)
"""

respuesta = """
PAMI ofrece los siguientes servicios para personas mayores:

1. Bolsón alimentario: Entrega mensual de productos secos seleccionados por nutricionistas.
2. Apoyo a la Dependencia y Fragilidad (PADyF): Apoyo económico para personas con limitaciones funcionales.
3. Viviendas en comodato: Otorgamiento de vivienda con descuento del 10% del haber previsional.
4. Club de día: Prestación socio comunitaria con jornadas de 6-8 horas diarias de actividades.

Requisitos generales:
- DNI
- Credencial de afiliación
- Evaluación del Equipo Social

Referencias:
- ID_SUB = 66 | SUBTIPO = 'Bolsón Alimentario-Probienestar-BCA'
- ID_SUB = 43 | SUBTIPO = 'Apoyo a la Dependencia y Fragilidad- PADyF'
"""

# Calcular los tokens
tokens_pregunta = contar_tokens(pregunta)
tokens_contexto = contar_tokens(contexto)
tokens_entrada = tokens_pregunta + tokens_contexto
tokens_salida = contar_tokens(respuesta)

print(f"Tokens de la pregunta: {tokens_pregunta}")
print(f"Tokens del contexto: {tokens_contexto}")
print(f"Tokens totales de entrada: {tokens_entrada}")
print(f"Tokens de la respuesta: {tokens_salida}")

# Probar la función de registro de tokens
log_message("Iniciando prueba de registro de tokens")
log_token_summary(tokens_entrada, tokens_salida, "gpt-4o-mini")
log_message("Finalizada prueba de registro de tokens") 