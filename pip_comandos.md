# Comandos de Instalación para Dependencias

Para instalar las dependencias necesarias para el conteo de tokens, debes ejecutar:

```bash
# Para instalar tiktoken (necesario para el conteo de tokens)
pip install tiktoken==0.6.0

# Para instalar datetime y json (generalmente ya vienen con Python)
# pip install datetime
# pip install json
```

Estos comandos deben ejecutarse en el entorno virtual donde está desplegada la aplicación. Si estás utilizando un entorno virtual específico, primero debes activarlo:

```bash
# En Windows
path\to\env\Scripts\activate

# En Linux/Mac
source path/to/env/bin/activate
```

## Nota importante

En las modificaciones realizadas, no se instaló ninguna dependencia nueva directamente en el entorno virtual. Se agregó el código que utiliza `tiktoken`, pero para que funcione correctamente, es necesario que esta biblioteca esté instalada en el entorno.

Si experimentas errores como:
```
ModuleNotFoundError: No module named 'tiktoken'
```

Deberás instalar la biblioteca utilizando:
```bash
pip install tiktoken==0.6.0
```

## Validación de la instalación

Para verificar que la biblioteca está correctamente instalada, puedes ejecutar:

```bash
python -c "import tiktoken; print('Tiktoken instalado correctamente')"
```

Si no aparece ningún error, la biblioteca está correctamente instalada. 