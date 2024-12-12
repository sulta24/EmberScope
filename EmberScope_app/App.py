import customtkinter as ctk
from PIL import Image, ImageTk
import requests
from io import BytesIO

# Функция для загрузки спутникового изображения
def fetch_satellite_image(lat, lon):
    try:
        # URL NASA GIBS для данных MODIS
        base_url = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
        params = {
            "SERVICE": "WMS",
            "REQUEST": "GetMap",
            "VERSION": "1.3.0",
            "LAYERS": "MODIS_Terra_CorrectedReflectance_TrueColor",
            "WIDTH": "512",
            "HEIGHT": "512",
            "FORMAT": "image/jpeg",
            "CRS": "EPSG:4326",
            "BBOX": f"{lat-0.5},{lon-0.5},{lat+0.5},{lon+0.5}"
        }

        response = requests.get(base_url, params=params)
        response.raise_for_status()

        return Image.open(BytesIO(response.content))
    except Exception as e:
        output_text.delete("1.0", ctk.END)
        output_text.insert(ctk.END, f"Ошибка загрузки изображения: {e}")
        return None

# Функция обработки данных
def process_data():
    try:
        lat = float(entry_lat.get())
        lon = float(entry_lon.get())

        # Загрузка спутникового изображения
        img = fetch_satellite_image(lat, lon)
        if img:
            img = img.resize((400, 400))
            bg_image = ImageTk.PhotoImage(img)
            map_label.configure(image=bg_image)
            map_label.image = bg_image

        output_text.delete("1.0", ctk.END)
        output_text.insert(ctk.END, f"Изображение загружено для координат:\nШирота: {lat}\nДолгота: {lon}")
    except ValueError as e:
        output_text.delete("1.0", ctk.END)
        output_text.insert(ctk.END, f"Ошибка: {str(e)}")

# Функции для автоматического ввода координат
def set_coordinates(lat, lon):
    entry_lat.delete(0, ctk.END)
    entry_lat.insert(0, lat)
    entry_lon.delete(0, ctk.END)
    entry_lon.insert(0, lon)

# Интерфейс приложения
app = ctk.CTk()
app.title("EmberScope")
app.geometry("800x600")

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Ввод данных
frame_input = ctk.CTkFrame(app)
frame_input.pack(pady=20, padx=20, fill="x")

ctk.CTkLabel(frame_input, text="Широта (lat):").grid(row=0, column=0, padx=5, pady=5)
entry_lat = ctk.CTkEntry(frame_input)
entry_lat.grid(row=0, column=1, padx=5, pady=5)

ctk.CTkLabel(frame_input, text="Долгота (lon):").grid(row=1, column=0, padx=5, pady=5)
entry_lon = ctk.CTkEntry(frame_input)
entry_lon.grid(row=1, column=1, padx=5, pady=5)

# Кнопки для местностей
button_evergreen_coniferous = ctk.CTkButton(app, text="Вечнозелёные хвойные леса", command=lambda: set_coordinates(60.0, -100.0))
button_evergreen_coniferous.pack(pady=5)

button_evergreen_broadleaf = ctk.CTkButton(app, text="Вечнозелёные широколиственные леса", command=lambda: set_coordinates(-3.4653, -62.2159))
button_evergreen_broadleaf.pack(pady=5)

button_deciduous_coniferous = ctk.CTkButton(app, text="Листопадные хвойные леса", command=lambda: set_coordinates(55.0, 90.0))
button_deciduous_coniferous.pack(pady=5)

button_deciduous_broadleaf = ctk.CTkButton(app, text="Листопадные широколиственные леса", command=lambda: set_coordinates(39.0, -77.0))
button_deciduous_broadleaf.pack(pady=5)

button_mixed_forests = ctk.CTkButton(app, text="Смешанные леса", command=lambda: set_coordinates(48.0, 8.0))
button_mixed_forests.pack(pady=5)

button_meadows = ctk.CTkButton(app, text="Луга", command=lambda: set_coordinates(45.0, 80.0))
button_meadows.pack(pady=5)

button_savannas = ctk.CTkButton(app, text="Саванны", command=lambda: set_coordinates(-2.3, 34.8))
button_savannas.pack(pady=5)

button_wetlands = ctk.CTkButton(app, text="Болотные территории", command=lambda: set_coordinates(25.5, -80.5))
button_wetlands.pack(pady=5)

button_shrublands = ctk.CTkButton(app, text="Кустарниковые территории", command=lambda: set_coordinates(34.0, -118.0))
button_shrublands.pack(pady=5)

button_deserts = ctk.CTkButton(app, text="Пустыни", command=lambda: set_coordinates(23.4162, 25.6628))
button_deserts.pack(pady=5)

button_agricultural_lands = ctk.CTkButton(app, text="Сельскохозяйственные земли", command=lambda: set_coordinates(41.0, -95.0))
button_agricultural_lands.pack(pady=5)

button_urban_areas = ctk.CTkButton(app, text="Городские территории", command=lambda: set_coordinates(40.7128, -74.0060))
button_urban_areas.pack(pady=5)

# Кнопка загрузки
button_process = ctk.CTkButton(app, text="Загрузить", command=process_data)
button_process.pack(pady=10)

# Карта
frame_map = ctk.CTkFrame(app, height=400)
frame_map.pack(pady=10, padx=20, fill="both", expand=True)

map_label = ctk.CTkLabel(frame_map, text="Здесь будет отображена карта", anchor="center")
map_label.pack(fill="both", expand=True)

# Вывод информации
output_text = ctk.CTkTextbox(app, height=100)
output_text.pack(pady=10, padx=20, fill="x")

# Запуск приложения
app.mainloop()
