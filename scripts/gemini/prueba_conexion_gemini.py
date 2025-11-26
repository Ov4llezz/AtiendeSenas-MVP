from google import genai
import os

# Mostrar que la API key está disponible
print("GEMINI_API_KEY =", os.environ.get("GEMINI_API_KEY"))

# Crear cliente
client = genai.Client()

# Llamar al modelo
result = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Di: 'Gemini funcionando correctamente en mi PC'."
)

print("Respuesta del modelo:")
print(result.text)


