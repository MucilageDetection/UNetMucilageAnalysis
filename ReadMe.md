# UNet Bölütleme

UNet 2015 yılında "U-Net: Convolutional Networks for Biomedical
Image Segmentation" adlı makalede önerilen bir evrişimsel sinir ağı tabanlı yöntemdir.

Yöntem kodlayıcı-çözücü yapısına sahiptir. Bu nedenle ilk katmanlarda girdi görüntüsünün boyutu pooling işlemleri ile sürekli olarak yarıya indirilir. Ardından çözücü bloklarda, düşük çözünürlükteki katmanlar büyültülerek girdi imgesi boyutuna getirilir.

Bu işlemin amacı girdinin uzamsal bilgisinin farklı kanallara yayılması ve imgenin bütününden yararlanılarak bir bölütleme yapılmasıdır. 

UNet yönteminin diğer kodlayıcı-çözücü yapılardan en önemli farkı, kodlayıcı ve çözücü bloklar arasında doğrudan bağlantı oluşturan "skip connection" bağlantılarını kullanmasıdır.

Bu sayede ağın eğitiminde ortaya çıkan vanishing gradient problemi çözülmekte ve kodlayıcı sırasında kaybedilen uzamsal çözünürlük geri kazanılmaktadır.

Bu ağ kullanılarak [Python](MucilageDetection.ipynb) dilinde bir kod yazılmıştır.

Eğitim için gerekli girdi görüntüleri, elle işaretlenen imgelerden kesilen $300 \times 300$ parçalar üzerinden yapılmıştır. Bu durumda test imgelerinde kullanılacak olan örtüşme penceresi aşağıdaki tabloya göre hesaplanmalıdır.

| Katman        | Büyüklük       |
| ------------- | -------------- |
| Girdi         | 256            |
| İlk Katman    | [252](#B1-4)   |
| Küçültme1     | [122](#B2/2-4) |
| Küçültme2     | [57](#B3/2-4) |
| Küçültme3     | [24.5](#B4/2-4)  |
| Küçültme4     | [8.25](#B5/2-4)  |
| Büyültme1     | [12.5](#B6*2-4)  |
| Büyültme2     | [21](#B7*2-4) |
| Büyültme3     | [38](#B8*2-4) |
| Büyültme4     | [72](#B9*2-4) |
| Geçerli Çıktı | [72](#B10)    |

| Katman        | Büyüklük       |
| ------------- | -------------- |
| Girdi         | 256            |
| İlk Katman    | [252](#B1)   |
| Küçültme1     | [122](#(B2-4)/2) |
| Küçültme2     | [57](#(B3-4)/2) |
| Küçültme3     | [24.5](#(B4-4)/2)  |
| Küçültme4     | [24.5](#(B5-4)/2)  |
| Büyültme1     | [45](#B6*2-4)  |
| Büyültme2     | [86](#B7*2-4) |
| Büyültme3     | [168](#B8*2-4) |
| Büyültme4     | [168](#B9) |
| Geçerli Çıktı | [168](#B10)    |

Yani bu hesaplamaya göre girdi imgeleri $300 \times 300$ kayan pencereden, $116 \times 116$ adım aralıkları ile kesilerek çıktı üretilmelidir.

Yazılan yöntem 35TPE görüntüleri üzerinde test edilmiş ve çıktıları aşağıda verilmiştir.


|                                TCI IMAGE                                 |                               MUCILAGE MAP                                |
| :----------------------------------------------------------------------: | :-----------------------------------------------------------------------: |
|                                2021-01-02                                |                                2021-01-02                                 |
| ![T35TPE_20210102T090351_20m](assets/TCI/T35TPE_20210102T090351_20m.jpg) | ![T35TPE_20210102T090351_20m](assets/UNet/T35TPE_20210102T090351_20m.jpg) |
|                                2021-02-06                                |                                2021-02-06                                 |
| ![T35TPE_20210206T090039_20m](assets/TCI/T35TPE_20210206T090039_20m.jpg) | ![T35TPE_20210206T090039_20m](assets/UNet/T35TPE_20210206T090039_20m.jpg) |
|                                2021-02-21                                |                                2021-02-21                                 |
| ![T35TPE_20210221T090001_20m](assets/TCI/T35TPE_20210221T090001_20m.jpg) | ![T35TPE_20210221T090001_20m](assets/UNet/T35TPE_20210221T090001_20m.jpg) |
|                                2021-02-23                                |                                2021-02-23                                 |
| ![T35TPE_20210223T084849_20m](assets/TCI/T35TPE_20210223T084849_20m.jpg) | ![T35TPE_20210223T084849_20m](assets/UNet/T35TPE_20210223T084849_20m.jpg) |
|                                2021-02-26                                |                                2021-02-26                                 |
| ![T35TPE_20210226T085829_20m](assets/TCI/T35TPE_20210226T085829_20m.jpg) | ![T35TPE_20210226T085829_20m](assets/UNet/T35TPE_20210226T085829_20m.jpg) |
|                                2021-03-03                                |                                2021-03-03                                 |
| ![T35TPE_20210303T085851_20m](assets/TCI/T35TPE_20210303T085851_20m.jpg) | ![T35TPE_20210303T085851_20m](assets/UNet/T35TPE_20210303T085851_20m.jpg) |
|                                2021-03-05                                |                                2021-03-05                                 |
| ![T35TPE_20210305T084739_20m](assets/TCI/T35TPE_20210305T084739_20m.jpg) | ![T35TPE_20210305T084739_20m](assets/UNet/T35TPE_20210305T084739_20m.jpg) |
|                                2021-03-28                                |                                2021-03-28                                 |
| ![T35TPE_20210328T085559_20m](assets/TCI/T35TPE_20210328T085559_20m.jpg) | ![T35TPE_20210328T085559_20m](assets/UNet/T35TPE_20210328T085559_20m.jpg) |
|                                2021-04-02                                |                                2021-04-02                                 |
| ![T35TPE_20210402T085551_20m](assets/TCI/T35TPE_20210402T085551_20m.jpg) | ![T35TPE_20210402T085551_20m](assets/UNet/T35TPE_20210402T085551_20m.jpg) |
|                                2021-04-14                                |                                2021-04-14                                 |
| ![T35TPE_20210414T084559_20m](assets/TCI/T35TPE_20210414T084559_20m.jpg) | ![T35TPE_20210414T084559_20m](assets/UNet/T35TPE_20210414T084559_20m.jpg) |
|                                2021-04-22                                |                                2021-04-22                                 |
| ![T35TPE_20210422T085551_20m](assets/TCI/T35TPE_20210422T085551_20m.jpg) | ![T35TPE_20210422T085551_20m](assets/UNet/T35TPE_20210422T085551_20m.jpg) |
|                                2021-04-29                                |                                2021-04-29                                 |
| ![T35TPE_20210429T084601_20m](assets/TCI/T35TPE_20210429T084601_20m.jpg) | ![T35TPE_20210429T084601_20m](assets/UNet/T35TPE_20210429T084601_20m.jpg) |
|                                2021-05-09                                |                                2021-05-09                                 |
| ![T35TPE_20210509T084601_20m](assets/TCI/T35TPE_20210509T084601_20m.jpg) | ![T35TPE_20210509T084601_20m](assets/UNet/T35TPE_20210509T084601_20m.jpg) |
|                                2021-05-14                                |                                2021-05-14                                 |
| ![T35TPE_20210514T084559_20m](assets/TCI/T35TPE_20210514T084559_20m.jpg) | ![T35TPE_20210514T084559_20m](assets/UNet/T35TPE_20210514T084559_20m.jpg) |
|                                2021-05-17                                |                                2021-05-17                                 |
| ![T35TPE_20210517T085559_20m](assets/TCI/T35TPE_20210517T085559_20m.jpg) | ![T35TPE_20210517T085559_20m](assets/UNet/T35TPE_20210517T085559_20m.jpg) |
|                                2021-05-19                                |                                2021-05-19                                 |
| ![T35TPE_20210519T084601_20m](assets/TCI/T35TPE_20210519T084601_20m.jpg) | ![T35TPE_20210519T084601_20m](assets/UNet/T35TPE_20210519T084601_20m.jpg) |
|                                2021-06-06                                |                                2021-06-06                                 |
| ![T35TPE_20210606T085559_20m](assets/TCI/T35TPE_20210606T085559_20m.jpg) | ![T35TPE_20210606T085559_20m](assets/UNet/T35TPE_20210606T085559_20m.jpg) |
|                                2021-06-11                                |                                2021-06-11                                 |
| ![T35TPE_20210611T085601_20m](assets/TCI/T35TPE_20210611T085601_20m.jpg) | ![T35TPE_20210611T085601_20m](assets/UNet/T35TPE_20210611T085601_20m.jpg) |
|                                2021-06-13                                |                                2021-06-13                                 |
| ![T35TPE_20210613T084559_20m](assets/TCI/T35TPE_20210613T084559_20m.jpg) | ![T35TPE_20210613T084559_20m](assets/UNet/T35TPE_20210613T084559_20m.jpg) |
|                                2021-06-26                                |                                2021-06-26                                 |
| ![T35TPE_20210626T085559_20m](assets/TCI/T35TPE_20210626T085559_20m.jpg) | ![T35TPE_20210626T085559_20m](assets/UNet/T35TPE_20210626T085559_20m.jpg) |
|                                2021-06-28                                |                                2021-06-28                                 |
| ![T35TPE_20210628T084601_20m](assets/TCI/T35TPE_20210628T084601_20m.jpg) | ![T35TPE_20210628T084601_20m](assets/UNet/T35TPE_20210628T084601_20m.jpg) |
|                                2021-07-01                                |                                2021-07-01                                 |
| ![T35TPE_20210701T085601_20m](assets/TCI/T35TPE_20210701T085601_20m.jpg) | ![T35TPE_20210701T085601_20m](assets/UNet/T35TPE_20210701T085601_20m.jpg) |
|                                2021-07-11                                |                                2021-07-11                                 |
| ![T35TPE_20210711T085601_20m](assets/TCI/T35TPE_20210711T085601_20m.jpg) | ![T35TPE_20210711T085601_20m](assets/UNet/T35TPE_20210711T085601_20m.jpg) |
|                                2021-07-13                                |                                2021-07-13                                 |
| ![T35TPE_20210713T084559_20m](assets/TCI/T35TPE_20210713T084559_20m.jpg) | ![T35TPE_20210713T084559_20m](assets/UNet/T35TPE_20210713T084559_20m.jpg) |
|                                2021-07-16                                |                                2021-07-16                                 |
| ![T35TPE_20210716T085559_20m](assets/TCI/T35TPE_20210716T085559_20m.jpg) | ![T35TPE_20210716T085559_20m](assets/UNet/T35TPE_20210716T085559_20m.jpg) |
|                                2021-07-28                                |                                2021-07-28                                 |
| ![T35TPE_20210728T084601_20m](assets/TCI/T35TPE_20210728T084601_20m.jpg) | ![T35TPE_20210728T084601_20m](assets/UNet/T35TPE_20210728T084601_20m.jpg) |
|                                2021-07-31                                |                                2021-07-31                                 |
| ![T35TPE_20210731T085601_20m](assets/TCI/T35TPE_20210731T085601_20m.jpg) | ![T35TPE_20210731T085601_20m](assets/UNet/T35TPE_20210731T085601_20m.jpg) |
|                                2021-08-02                                |                                2021-08-02                                 |
| ![T35TPE_20210802T084559_20m](assets/TCI/T35TPE_20210802T084559_20m.jpg) | ![T35TPE_20210802T084559_20m](assets/UNet/T35TPE_20210802T084559_20m.jpg) |
|                                2021-08-05                                |                                2021-08-05                                 |
| ![T35TPE_20210805T085559_20m](assets/TCI/T35TPE_20210805T085559_20m.jpg) | ![T35TPE_20210805T085559_20m](assets/UNet/T35TPE_20210805T085559_20m.jpg) |
|                                2021-08-10                                |                                2021-08-10                                 |
| ![T35TPE_20210810T085601_20m](assets/TCI/T35TPE_20210810T085601_20m.jpg) | ![T35TPE_20210810T085601_20m](assets/UNet/T35TPE_20210810T085601_20m.jpg) |
|                                2021-08-15                                |                                2021-08-15                                 |
| ![T35TPE_20210815T085559_20m](assets/TCI/T35TPE_20210815T085559_20m.jpg) | ![T35TPE_20210815T085559_20m](assets/UNet/T35TPE_20210815T085559_20m.jpg) |
|                                2021-08-17                                |                                2021-08-17                                 |
| ![T35TPE_20210817T084601_20m](assets/TCI/T35TPE_20210817T084601_20m.jpg) | ![T35TPE_20210817T084601_20m](assets/UNet/T35TPE_20210817T084601_20m.jpg) |
|                                2021-08-25                                |                                2021-08-25                                 |
| ![T35TPE_20210825T085559_20m](assets/TCI/T35TPE_20210825T085559_20m.jpg) | ![T35TPE_20210825T085559_20m](assets/UNet/T35TPE_20210825T085559_20m.jpg) |
|                                2021-08-27                                |                                2021-08-27                                 |
| ![T35TPE_20210827T084601_20m](assets/TCI/T35TPE_20210827T084601_20m.jpg) | ![T35TPE_20210827T084601_20m](assets/UNet/T35TPE_20210827T084601_20m.jpg) |
|                                2021-08-30                                |                                2021-08-30                                 |
| ![T35TPE_20210830T085601_20m](assets/TCI/T35TPE_20210830T085601_20m.jpg) | ![T35TPE_20210830T085601_20m](assets/UNet/T35TPE_20210830T085601_20m.jpg) |