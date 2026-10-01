# Ampliación aprobada: TabPFN-3 y TabPFN-3.5

El usuario pidió evaluar ambas versiones después de aceptar sus licencias el 1 de octubre de 2026. TabPFN-3 conserva el experimento original; TabPFN-3.5 se agrega como diagnóstico separado y no lo reemplaza.

Se conservan D_month_index, los cortes agosto→septiembre, septiembre→octubre y octubre→noviembre, todas las filas de validación y Gini mensual/media simple rolling. Cada versión usa su selector explícito (V3 o V3_5) y defaults publicados, sin búsqueda ni muestreo externo. Se registran el contexto interno, checkpoint, configuración, código y hashes por fold.

El módulo original de foundation permanece congelado para conservar la identidad de los resultados TabFM existentes. foundation35.py usa las mismas utilidades de protocolo y linaje con una identidad independiente y rechazo de checkpoints de otra versión, incluidos los Fast.

Se prueba cada versión localmente; ante falta de memoria o 30 minutos por fold, se migra a Colab. Se recupera y verifica cada fold completo antes de continuar. La clave del .env se transmite únicamente en el entorno o un archivo temporal privado, fuera de los snapshots y logs. .env queda excluido de Git.

Referencia oficial: https://huggingface.co/Prior-Labs/tabpfn_3_5
