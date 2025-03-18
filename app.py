from flask import Flask, render_template, request
import logging

# Configurar logging para guardar toda la salida en un archivo de log
logging.basicConfig(filename='app_log.log', level=logging.DEBUG, format='%(asctime)s - %(message)s')

# Crear un logger para este módulo
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['DEBUG'] = False

try:
   
    from grafo_AGENTE_SERV_flask import process_question as process_question_servicios  
    
    logger.info("Módulos importados correctamente")
except ImportError as e:
    logger.error(f"Error al importar módulos: {e}")
    
@app.route('/')
def home():
    return render_template('inicio.html')



@app.route('/servicios-simap', methods=['GET', 'POST'])
def servicios_simap():
    resultado = ""
    pregunta = ""
    fecha_desde = "2024-01-01"
    fecha_hasta = "2024-12-31"
    k = 50

    if request.method == 'POST':
        pregunta = request.form['pregunta']
        fecha_desde = request.form.get('fecha_desde', "2024-01-01")
        fecha_hasta = request.form.get('fecha_hasta', "2024-12-31")
        k = int(request.form.get('k', 50))
        try:
            logger.info(f"Procesando pregunta de servicios: {pregunta}")
            resultado = process_question_servicios(pregunta, fecha_desde, fecha_hasta, k)
            logger.info(f"Resultado obtenido: {resultado}")
        except Exception as e:
            logger.error(f"Error al procesar la pregunta de servicios: {str(e)}")
            resultado = f"Error al procesar la pregunta: {str(e)}"

    return render_template('servicios_simap.html', resultado=resultado, pregunta=pregunta, fecha_desde=fecha_desde, fecha_hasta=fecha_hasta, k=k)




@app.route('/noticias-simap', methods=['GET', 'POST'])
def noticias_simap():
    resultado = ""
    if request.method == 'POST':
        pregunta = request.form['pregunta']
        fecha_desde = request.form['fecha_desde']
        fecha_hasta = request.form['fecha_hasta']
        k = int(request.form['k'])
        try:
            logger.info(f"Procesando pregunta de noticias: {pregunta}")
            resultado = process_question_noticias(pregunta, fecha_desde, fecha_hasta, k)
            logger.info(f"Resultado obtenido: {resultado}")
        except Exception as e:
            logger.error(f"Error al procesar la pregunta de noticias: {str(e)}")
            resultado = f"Error al procesar la pregunta: {str(e)}"
    return render_template('noticias_simap.html', resultado=resultado)

@app.route('/resoluciones-simap', methods=['GET', 'POST'])
def resoluciones_simap():
    resultado = ""
    if request.method == 'POST':
        pregunta = request.form['pregunta']
        fecha_desde = request.form['fecha_desde']
        fecha_hasta = request.form['fecha_hasta']
        k = int(request.form['k'])
        try:
            logger.info(f"Procesando pregunta de resoluciones: {pregunta}")
            resultado = process_question_resoluciones(pregunta, fecha_desde, fecha_hasta, k)
            logger.info(f"Resultado obtenido: {resultado}")
        except Exception as e:
            logger.error(f"Error al procesar la pregunta de resoluciones: {str(e)}")
            resultado = f"Error al procesar la pregunta: {str(e)}"
    return render_template('resoluciones_simap.html', resultado=resultado)

@app.route('/extracto_resoluciones', methods=['GET', 'POST'])
def extracto_resoluciones():
    if request.method == 'POST':
        fecha_desde = request.form['fecha_desde']
        fecha_hasta = request.form['fecha_hasta']
        k = int(request.form['k'])
        nivel = request.form['nivel']
        pregunta = request.form['pregunta']
        
        resultado = process_question_bole(pregunta, fecha_desde, fecha_hasta, k, nivel)
        return render_template('extracto_resoluciones.html', resultado=resultado)
    
    return render_template('extracto_resoluciones.html')

@app.route('/instructo', methods=['GET', 'POST'])
def instructo():
    resultado = ""
    if request.method == 'POST':
        pregunta = request.form['pregunta']
        fecha_desde = request.form['fecha_desde']
        fecha_hasta = request.form['fecha_hasta']
        k = int(request.form['k'])
        try:
            logger.info(f"Procesando pregunta de instrucciones: {pregunta}")
            resultado = process_question_instrucciones(pregunta, fecha_desde, fecha_hasta, k)
            logger.info(f"Resultado obtenido: {resultado}")
        except Exception as e:
            logger.error(f"Error al procesar la pregunta de instrucciones: {str(e)}")
            resultado = f"Error al procesar la pregunta: {str(e)}"
    return render_template('instructo.html', resultado=resultado)

@app.route('/externo', methods=['GET', 'POST'])
def externo():
    resultado = ""
    if request.method == 'POST':
        pregunta = request.form['pregunta']
        fecha_desde = request.form['fecha_desde']
        fecha_hasta = request.form['fecha_hasta']
        k = int(request.form['k'])
        
        try:
            logger.info(f"Procesando pregunta externa: {pregunta}")
            resultado = process_question_extracto(pregunta, fecha_desde, fecha_hasta, k)
            logger.info(f"Resultado obtenido: {resultado}")
        except Exception as e:
            logger.error(f"Error al procesar la pregunta externa: {str(e)}")
            resultado = f"Error al procesar la pregunta: {str(e)}"
    
    return render_template('externo.html', resultado=resultado)

@app.route('/todo', methods=['GET', 'POST'])
def todo():
    resultado = ""
    if request.method == 'POST':
        pregunta = request.form['pregunta']
        fecha_desde = request.form['fecha_desde']
        fecha_hasta = request.form['fecha_hasta']
        k = int(request.form['k'])
        try:
            logger.info(f"Procesando pregunta de todo: {pregunta}")
            resultado = process_question_todo(pregunta, fecha_desde, fecha_hasta, k)
            logger.info(f"Resultado obtenido: {resultado}")
        except Exception as e:
            logger.error(f"Error al procesar la pregunta de todo: {str(e)}")
            resultado = f"Error al procesar la pregunta: {str(e)}"
    return render_template('todo.html', resultado=resultado)

"""if __name__ == '__main__':
    logger.info("Iniciando la aplicación Flask...")
    app.run(debug=False)
"""
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)