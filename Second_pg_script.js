let fireInterval; // Переменная для хранения интервала (цикла)
let fireCenter; // Переменная для центра огня
let spreadInterval = 200; // Интервал между добавлением точек (в миллисекундах)
let fireDirection = 90; // Направление огня (градусы)

// Функция для отображения изображения
function showImage(landscape) {
  const image = document.getElementById("image");
  const canvas = document.getElementById("fireCanvas");

  // Путь к изображениям
  const imagePath = `images/${landscape}.png`;

  // Изменяем источник изображения
  image.src = imagePath;
  image.alt = landscape;

  // Когда изображение загружено, настроим холст
  image.onload = function () {
    canvas.width = image.width;
    canvas.height = image.height;

    image.classList.add("active");
  };
}

// Функция для начала цикла огня
function startFireCycle(landscape) {
  const canvas = document.getElementById("fireCanvas");
  const ctx = canvas.getContext("2d");

  // Показываем изображение
  showImage(landscape);

  // Очищаем предыдущие точки огня
  if (fireInterval) {
    clearInterval(fireInterval);
  }

  // Устанавливаем начальный центр огня (середина изображения)
  fireCenter = {
    x: canvas.width / 2,
    y: canvas.height / 2,
  };

  // Устанавливаем радиус как 10% от меньшей стороны изображения, но не больше 100px
  const spreadRadius = Math.min(Math.min(canvas.width, canvas.height) * 0.1, 100);

  // Запускаем цикл для постепенного распространения огня
  fireInterval = setInterval(() => {
    fireCenter.x += Math.cos((fireDirection) * Math.PI / 180) * 5;
    fireCenter.y += Math.sin((fireDirection - 180) * Math.PI / 180) * 5;

    spreadFire(ctx, canvas.width, canvas.height, fireCenter.x, fireCenter.y, spreadRadius);
  }, spreadInterval);
}

// Функция для распространения огня
function spreadFire(ctx, canvasWidth, canvasHeight, startX, startY, radius) {
  const firePointsCount = 5;

  for (let i = 0; i < firePointsCount; i++) {
    const angle = Math.random() * 2 * Math.PI;
    const distance = Math.random() * radius;

    // Генерация координат
    const x = startX + distance * Math.cos(angle);
    const y = startY + distance * Math.sin(angle);

    // Размер точек: 0.5-2% от меньшей стороны экрана
    const screenSize = Math.min(window.innerWidth, window.innerHeight);
    const size = Math.random() * Math.max(Math.min(screenSize * 0.01, 100), 3);

    // Убедимся, что точка находится в пределах изображения
    if (x >= 0 && x <= canvasWidth && y >= 0 && y <= canvasHeight) {
      ctx.beginPath();
      ctx.arc(x, y, size, 0, Math.PI * 2);
      ctx.fillStyle = "red";
      ctx.fill();
    }
  }
}

// Функция для обновления параметров огня
function updateFireCycle() {
  const intervalInput = document.getElementById("interval");
  const directionInput = document.getElementById("direction");

  spreadInterval = 1000 - intervalInput.value;
  fireDirection = parseInt(directionInput.value, 10);

  if (fireInterval) {
    clearInterval(fireInterval);
  }

  alert(`Параметры обновлены:
- Интервал между точками: ${spreadInterval} миллисекунд
- Направление огня: ${fireDirection} градусов`);

  const image = document.getElementById("image");
  if (image.src) {
    startFireCycle(image.alt);
  }
}
