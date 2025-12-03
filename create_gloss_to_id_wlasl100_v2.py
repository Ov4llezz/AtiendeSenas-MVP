import json

# Cargar nslt_100.json
with open('data/wlasl100_v2/nslt_100.json', 'r') as f:
    video_data = json.load(f)

# Cargar WLASL_v0.3.json como lista de glosas
with open('data/wlasl100_v2/WLASL_v0.3.json', 'r') as f:
    wlasl_data = json.load(f)

# Crear mapeo de gloss_name -> índice (el índice será el gloss_id)
wlasl_glosses_list = [item['gloss'] for item in wlasl_data]

# Recopilar todos los gloss_ids únicos usados en nslt_100.json
used_gloss_ids = set()
for video_id, data in video_data.items():
    if 'action' in data and len(data['action']) > 0:
        gloss_id = data['action'][0]
        used_gloss_ids.add(gloss_id)

# Crear el mapeo gloss_name -> gloss_id solo para las 100 clases usadas
gloss_to_id = {}
for gloss_id in sorted(used_gloss_ids):
    if gloss_id < len(wlasl_glosses_list):
        gloss_name = wlasl_glosses_list[gloss_id]
        gloss_to_id[gloss_name] = gloss_id
    else:
        gloss_to_id[f"unknown_{gloss_id}"] = gloss_id

# Guardar el archivo
with open('data/wlasl100_v2/gloss_to_id.json', 'w') as f:
    json.dump(gloss_to_id, f, indent=2)

print(f"Archivo gloss_to_id.json creado con {len(gloss_to_id)} glosas")
print(f"Glosas: {list(gloss_to_id.keys())[:10]}...")
