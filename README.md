
# Задание:
Необходимо натренировать классификатор открытых/закрытых глаз, используя заданную обучающую выборку. Обучающая выборка состоит из 4000 черно-белых изображений размером 24х24.
# Анализ задачи
Учитывая необходимость бинарной классификации и простоту изображений, количества предоставленных данных достаточно для решения этой задачи. В ходе этой работы могут быть протестированы различные архитектуры, а также техника Transfer Learning with Pre-trained Models для выбора наилучшей производительности:\
**1. Convolutional Neural Networks (CNNs)**\
Задача классификации небольших изображений хорошо изучена и классические модели, которые представлены в библиотеку pytorch, вполне способны успешно ее решить.
 	
   - **VGGNet[1]:** Одна из первых моделей после LeNet, в которую добавили больше слоев и уменьшили размер свертки. Для небольших изображений можно использовать более компактные варианты, например VGG-11 или VGG-16.
   - **ResNet[2]:** Архитектуры ResNet вводят пропускные связи и могут использоваться для более глубоких сетей без проблемы исчезающего градиента.
   - **EfficientNet[3]:** Более современная архитектура, которая хорошо масштабируется и эффективна с точки зрения соотношения точности и вычислительных затрат.
   - **YOLOv8[4]:** современная модель с удобным API, которая объединяет высокую скорость и точность обнаружения объектов на изображении или видео.
   - **Mobilenet[5]:** легкая нейронная сеть, оптимизированная для мобильных устройств, которая также обеспечивает высокую скорость работы и хорошее качество детектирования объектов.

**2. Transfer Learning with Pre-trained Models**
   - 	Предварительно обученные CNN: Также стоит протестировать уже предобученные модели. Несмотря на то, что они обучены на другом domain, модели часто фиксируют полезные признаки, которые позволят немного улучшить качество распознавания на текущей задаче. В библиотеку pytorch модели представлены как без весов, так и обученные на датасете ImageNet.
   - Модель в качестве экстрактора признаков + ML: Предварительно обученная модель в качестве экстрактора признаков может послужить для создания пространства эмбедингов, которое можно будет классифицировать при помощи SVM классификатора или кластеризации.
   - Zero-shot classification: CLIP[6], DINOv2[7]
# Подготовка данных
Размечать вручную 4к изображений слишком трудоемко, поэтому я решила использовать semi-supervised подход. При помощи инструмента CVAT были размечены 100 изображений, из которых 80 в последующем использовались для тренировки модели-разметчика. Так как это кол-во данных слишком маленькое для качественного обучения, было решено обучить две модели ResNet50 и YOLOv8 (данные о процессе обучения представлены на рис.1). Если предсказания об одном и том же изображении совпадают, то изображение переносится в папку соответственно “0” или “1”, потому что оно с большей вероятностью классифицировано правильно. Если предсказания разнятся, то этот файл переносился в папку “unclear” для полноценной ручной классификации.  

 
(а)
  
                                                                                (б)                                      тренировочная выборка
Рисунок 1. Тренировочный и валидационный лосс, метрика точности при обучении моделей ResNet 50(a) и YOLOv8(б)

В результате получилось реалистичное распределение по классам, где 0 - 'close', 1 -'open': Resnet (0 – 1743 изображений, 1 - 2257), YOLOv8 (1 – 1866, 0 – 2134). Визуальный анализ, представленный на изображении 2, подтвердил, что разметка получилась достаточно хорошей для дальнейшей нетрудоемкой ручной обработки. После сравнения предсказаний получилось 1213 несовпадений из которых 473 принадлежит классу 1. 
  
Рисунок 2. Пример визуального анализа с красными рамками, где обе модели ошиблись при классификации
В ходе ручной сортировке были обнаружены следующие особенности:
•	Кол-во False Positive изображений больше, чем False Negative. 
•	Существует много изображений с полузакрытыми и узкими глазами (примеры на рисунке 3), которые неоднозначно поддаются классификации, что может вызвать неправильные предсказания при тестировании на датасете с другой логикой разметки (не хватает промежуточного класса). 

                
Рисунок 3. Примеры изображений, которые были классифицированы модельно как “1”, однако при этом некоторые из них принадлежат к классу “0”
После проведения ручной сортировки в финальном датасете в 1 классе стало 1823 изображений, а во 2 – 2171.  В первом классе на 16% меньше данных, что не является критическим для сбалансированности датасета.
# Аугментации изображений

В ходе ручной сортировки было удобно проанализировать данные и подобрать список необходимых аугментаций из библиотеки pytorch. Также стоит учитывать, что все изображения уже являются кропом лица и захватывают приблизительно одинаковую область, где центрирование произведено относительно центра глаза [8]. На рисунке 4 приведена визуализация данных из train dataloader with augmentation и из test dataloader.
v2.RandomHorizontalFlip(p=0.5),
v2.ColorJitter(brightness=0.5, contrast=1),
v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
v2.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
v2.RandomAdjustSharpness(sharpness_factor=0, p=0.3), 

  
(а)                                                                        (б)
Рисунок 4. Изображения с аугментациями из train dataset (a) и test dataset (б) без аугментаций


 
N
# Обучение моделей
В качестве начального эксперимента была протестирована модель CLIP с архитектурой энкодера ResNet50 и результаты показали, что она не походит для этой задачи.  На неразмеченных данных было обнаружено только 12 изображений первой категории из 4000, что совершенно неправдоподобно. Модель CLIP была обучена на RGB-изображениях, поэтому ее работа на черно-белых изображениях может быть не оптимальной. Адаптация к полутоновым изображениям потребует дополнительных действий, таких как преобразование полутоновых изображений в RGB или модификация модели для работы непосредственно с одноканальными изображениями. Если делать fine-tuning этой модели, то могут также возникнуть проблемы из-за небольшого кол-ва тренировочных данных и невозможностью заморозить веса для энкодера.
Тестовый и тренировочный датасет разделены в соотношении: 20/80. Для сравнения производительности классических моделей будут использоваться одинаковые параметры обучения: N_EPOCHS = 30, opt = torch.optim.Adam(gray_model.parameters(), lr=0.001), criterion = torch.nn.BCELoss(). Также для вывода дискретных значений от 0 до 1 потребовалось изменить параметры финальное слоя/классификатора модели, чтобы предсказывалось только одно число (рисунок 5). 
 
Рисунок 5. Модификация классификатора для модели Resnet18
Train/test loss, train/test accuracy для всех моделей, кроме YOLOv8 (рисунок 6) доступны по ссылке на проект в wandb:
 https://wandb.ai/anna-ilyushina20/Eyes_open_close_classification?nw=nwuserannailyushina20
 
Рисунок 6. Train/test loss при обучении модели YOLOv8-n c гиперпараметрами batch = 128, epochs=30, imgsz=24, workers=2, warmup_epochs = 5, lr0= 0.005






# Тестирование
Результаты тестирования приведены в таблице 1, где помимо основных метрик отражено кол-во параметров для обучения в нейросети и EER (Equal error rate). Для классификации был выбран параметр treashold = 0.6, при прохождении которого предсказание засчитывалось за положительное. 
Таблица 1. Параметры и метрики полученные в результате обучения моделей, Pretrain = True
Модель	Precision	Recall	EER	Params
(M)
Resnet18	0.94	0.94	0.05	11.6
Resnet50	0.96	0.92	0.027	25.6
Resnet101	0.94	0.88	0.046	44.5
mobilenet_v3_small	0.95	0.93	0.041	2.5
mobilenet_v3_large	0.95	0.93	0.041	5.4
YOLOv8 - n	0.88	0.96	0.1	2.7
YOLOv8 - m	0.91	0.97	0.08	17.0
Результаты показали, что наилучшую производительность имеет модель Resnet50, а на втором месте Mobilenet v3 small. На мобильных устройствах или в средах с ограниченными ресурсами предпочтительнее выбрать MobileNetV3 Small, потому что она имеет меньшее количество параметров и более компактный размер. При этом, модель со слишком большим кол-вом параметров хуже классифицирует простые изображения, что связано с низкой способностью обобщения из-за чрезмерного запоминания всех шумов и особенностей данных. YOLOv8 показала наихудший результат, что связано с ее специализацией на задачу детекции, а также меньшей глубиной архитектуры и кол-вом параметром по сравнению с Resnet50. Также стоит отметить, что обучение YOLOv8 заняло существенно больше времени по сравнению с остальными моделями.  
Classification report (рисунок 7) подтверждает отсутствие дисбаланса классов, так как оба класса предсказываются примерно с одинаковой точностью. Из 1 класса были верно классифицированы TP - 416 изображений, а 18 - FN. Из 0 класса 338 – TN и 26 – FP. Таким образом, модель склонна больше классифицировать закрытые как открытые. Информация о тестировании моделей семейства Resnet и mobilenet говорили о схожей ситуации, а вот YOLO наоборот, склонна классифицировать открытые как закрытые.
  
(а)                                                                        (б)
Рисунок 7. Сlassification Report and Confusion Matrix о тестировании модели mobilenet_v3_large(a) и YOLOv8-n(б)
На рисунке 8 показаны предсказания в диапазоне от 0 до 1 для простых изображений из тестовой выборки с пользованием модели Resnet50. Анализ более сложных случаев False Positive(FP) and False Negative(FN) показывает, что модель обладает хорошими обобщающими способностями, так как в некоторых случаях предсказывает правильнее, чем разметка (ошибка в разметке отмечена красной рамкой отмечена на рисунке 9). Те граничные случаи полузакрытых глаз, о которых было написано в разделе подготовки данных, ожидаемо оказались сложными для предсказания. В таком случае, я бы сделала вывод, что для открытых глаз treashold можно поднять до 0.8~0.9, а для отметки полузакрытых оставить 0.3-0.7.

 
Рисунок 8. Визуализация изображений с предсказанием с отображением истинного класса используя Resnet50
  
(а)                                                                        (б)
Рисунок 9. Предсказания для изображений, которые были отнесены в категорию FP (a) и FN (б) 
Дальнейшая работа, которая не была бы ограничена временными ограничениями Google Collab и превышением квоты загрузки с Google Disk могла бы быть следующей:
•	Проведение поиска параметров аугментаций
•	Проведение поиска гиперпараметров обучения модели вручную или настроить, например, библиотеку optuna.
•	Проанализировать скорость работы моделей и в зависимости от особенной задачи(offline/online) выбрать наиболее подходящую по компромиссу между скоростью и производительностью.
•	Собрать больше данных о полузакрытых глазах и вынести это в отдельный класс. Также могу предположить, что помимо визуальной информации о самом глазе можно учитывать информацию о лице целиком, что поможет уточнить открыты или закрыты глаза.
Тестирование других более сложных методов классификации, таких как отображение изображения в векторное пространство эмбедингов или использование трансформеров для решения этой задачи не целесообразно, так как они потребуют большое кол-во тренировочных данных и вычислительных ресурсов (предсказание также будет занимать больше времени). Задача успешно решается с помощью классических CV моделей и полученная точность является достаточно хорошей, чтобы продолжать с ними работать и оптимизировать гиперпараметры.
# Список Литературы
[1] Simonyan K., Zisserman A. Very deep convolutional networks for large-scale image recognition //arXiv preprint arXiv:1409.1556. – 2014.\
[2] He K. et al. Deep residual learning for image recognition //Proceedings of the IEEE conference on computer vision and pattern recognition. – 2016. – С. 770-778.\
[3] Tan M. Efficientnet: Rethinking model scaling for convolutional neural networks //arXiv preprint arXiv:1905.11946. – 2019.\
[4] Redmon J. You only look once: Unified, real-time object detection //Proceedings of the IEEE conference on computer vision and pattern recognition. – 2016.\
[5] Sandler M. et al. Mobilenetv2: Inverted residuals and linear bottlenecks //Proceedings of the IEEE conference on computer vision and pattern recognition. – 2018. – С. 4510-4520.\
[6] Radford A. et al. Learning transferable visual models from natural language supervision //International conference on machine learning. – PMLR, 2021. – С. 8748-8763.\
[7] Oquab M. et al. Dinov2: Learning robust visual features without supervision //arXiv preprint arXiv:2304.07193. – 2023.\
[8] Wang J., Lee S. Data augmentation methods applying grayscale images for convolutional neural networks in machine vision //Applied Sciences. – 2021. – Т. 11. – №. 15. – С. 6721.

