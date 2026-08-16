# Crop Geometry Lab

Crop Geometry Lab, hava fotoğraflarındaki ekin çemberlerini insan denetimli ve
ölçülebilir geometrik veriye dönüştüren, yalnızca yerelde çalışan bir analiz
aracıdır. İlk çalışma görüntüsü Milk Hill Galaxy Spiral 2001’dir.

Bu sürümün bilimsel sınırı güvenilir geometrik veri çıkarımı ve altılı dönel
simetri ölçümüdür. Torus yorumu, fiziksel neden çıkarımı ve yapay zekâ tabanlı
super-resolution kapsam dışındadır.

## Özellikler

### Geri alınabilir Ön İşleme çalışma alanı

- JPG, PNG ve WebP yükleme; görüntü üzerinde yakınlaştırma ve kaydırma
- Çokgen analiz bölgesi
- Boyanabilir dışlama maskesi; silgi, geri alma, görünürlük ve opaklık
- Dört noktalı manuel perspektif düzeltme; varsayılan olarak kaynak görüntünün
  tamamını aynı boyutlu çıktı tuvaline sığdıran kırpmasız dönüşüm
- Yoğun yön alanı, çizgi desteği ve periyodik aralık uzlaşmasından kararlı
  tarla izi/perspektif önerisi
- Otomatik veya px cinsinden kilitli iz aralığı ve ayarlanabilir düzenlilik
- Önerilen izleri seçme, taşıma, iki ucundan düzeltme, silme, yeni iz çizme,
  topluca paralelleştirme ve geri alma
- Düzenlenmiş öneriyi görüntüleme, reddetme veya açıkça onaylama
- LAB parlaklık kanalında büyük ölçekli aydınlatma tahmini
- Ayarlanabilir arka plan çıkarma/bölme ve CLAHE
- `flattened_crop`, `standing_crop`, `tramline` ve `shadow` renk örnekleri
- Örneklerden renk tabanlı ön plan olasılık haritası
- Ayarlanabilir genişlikte tarla izi dışlama maskesi
- Açılıp kapatılabilir, sıralı ve non-destructive işlem yığını
- Her işlem için önce/sonra görünümü
- `preprocess_profile.json` kaydetme ve yeniden yükleme

Kaynak görüntü hiçbir zaman üzerine yazılmaz. Tarla izleri ölçüm görüntüsünden
silinmez ve inpaint edilmez; yalnızca algılamaya verilmeyen dışlama maskesine
eklenir. Görüntü büyütme yalnızca tarayıcı tuvalindeki görünüm içindir.

### Perspektif Kalitesi

- Uygulanmamış, önerilmiş veya onaylanmış dört nokta/tramline dönüşümünü ham
  görüntüye ya da anotasyona yazmadan değerlendirme
- Tramline açı sapması, aralık değişim katsayısı ve 0–100 paralellik puanı
- Tüm ana çemberler, en büyük 18 ana çember veya seçili çemberler için
  önce/sonra elips oranı, dairesellik kalıntısı ve iyileşme puanı
- Altı kol için yarıçap, merkez uzaklığı, açı ve spiral uyum tutarlılığı
- K5–K6 sıra 1…13 yarıçap/uzaklık/normalize çap farkı karşılaştırması
- Homografi alan değişimi, shear, anizotropi ve yerel bozulma cezası
- Düzenlenebilir tramline, dairesellik, kol simetrisi, K5–K6 ve warp ağırlıkları
- Önce/sonra, yan yana, elips, tramline ve kol ölçek farkı görünümleri
- Kaynak/düzeltilmiş koordinat uzayı seçimi ve seçili çember yakın görünümü
- JSON, üç CSV ve karşılaştırma/overlay PNG dışa aktarımı

Eksik yapısal veri bir bileşeni tahmin ettirmez; ilgili değer panelde `—`
olarak gösterilir ve güven seviyesi veri kapsamına göre düşürülür. Modül
çemberleri taşımaz, etiketleri değiştirmez ve S-sapması düzeltmesi uygulamaz.

### Boyuta duyarlı daire algılama

- Kullanıcının yan dairelerde ortak kullanılan küçük, orta ve büyük
  standartlar için örnek daire seçmesi
- Yan büyük standardında Hough Circle ve ellipse fitting
- Yan orta standardında kontur ve dairesellik analizi
- Yan küçük standardında LoG/DoG blob algılama
- Üç yan standardından bağımsız min–maks aralığında değişken çaplı ana kol
  geçişi; daha sıkı ve ayrı güven/Hough eşikleri
- Örneklerden türetilen yan standart aralıklarının ana kol geçişinden
  dışlanması ve iki hattın ortak aday birleştirmesinde tekilleştirilmesi
- Yan standart hattından gelen adayların başlangıçta `satellite`, bağımsız
  ana kol hattından gelenlerin `main` olarak işaretlenmesi; iki hattın aynı
  geometriye düştüğü durumda algılayıcı-aile uzlaşmasıyla rol seçimi
- Temas eden bölgeler için isteğe bağlı distance transform + watershed
- Örtüşen parçalar üzerinde, kaynak ayrıntı üretmeden çok ölçekli algılama
- Merkez ve yarıçap yakınlığına göre algılayıcı sonuçlarını birleştirme
- Uzamsal indeksli aday birleştirme ve değişmeyen ön işleme sonucunu tekrar kullanma
- 409 kayıtlı Milk Hill denetimine alan bırakan 500 varsayılan aday sınırı;
  sınırı görüntü konumuna göre değil en yüksek güven puanına göre uygulama
- Aday başına:
  `edge_score`, `color_score`, `circularity_score`,
  `detector_agreement`, `symmetry_support` ve `final_confidence`
- Onaylanan daireyi kilitleme; sonraki turlarda kilitli geometrinin korunması
- LAB parlaklık/kenar desteği en düşük adayı yalnızca insan incelemesi için
  seçip yakınlaştırma; otomatik silme yapılmaz

Otomatik sonuçlar yalnızca düzenlenebilir adaylardır. Manuel ekleme, düzeltme,
silme ve kilitleme otomatik sonuçlardan her zaman önceliklidir.
Kalabalık görüntülerde etiketler varsayılan olarak kapalıdır; istenirse altılı
analiz bölümünden açılabilir.

### Ölçüm ve dışa aktarma

- Daire ekleme, silme, merkez taşıma ve yarıçap değiştirme
- Ana/yan çember, parent, kol, sıra ve taraf alanlarını düzenleme
- Daire başına çizgi rengi, piksel kalınlığı, dolgu rengi ve dolgu opaklığı
- Otomatik algılama güven eşiğini 0,01 adımlarla ayarlama
- Genel merkeze göre uzaklık ve açı
- Düz/radyal veya yarıçapla dönen Archimedean spiral kol modeli
- Güçlü dairelerden faz ve `derece / 100 px` spiral dönüş önerisi; yalnızca
  kullanıcı onayıyla kontrollere uygulanır
- Elle değiştirilen kol/sıra/yan alanlarının sonraki analizlerde korunması
- Kol değişiminde eski sıra değerini geçersiz kılma ve manuel kol atamalarını
  koruyarak her kolu merkezden dışarı `1…N` yeniden numaralama
- 60°, 120°, 180°, 240° ve 300° görüntü/geometri simetri hataları
- Altı kol için ayrı renkli katman
- Daireler, manuel alan kilitleri, silme günlüğü, merkez, kol modeli ve ön
  işleme profilini içeren `.work.json` çalışma kaydı ve geri yükleme
- Katmanlı sonuç PNG’si, CSV ve JSON
- Perspektif uygulanırsa kaynak/düzeltilmiş merkez, yarıçap ve elips eksenleri
- Beş denetim çıktısı:
  `rectified.png`, `illumination_corrected.png`,
  `foreground_probability.png`, `exclusion_mask.png` ve
  `detection_overlay.png`

### İnsan onaylı yapısal sınıflandırma

- `center`, `main`, `satellite` ve `unclassified` çember türleri; eski
  `side` kayıtlarını okumaya devam eden geriye uyumluluk
- Her kol için Toplam / Ana / Yan / Sınıfsız ve hedef farkını gösteren
  **Kol Denetimi** paneli
- 68/13/55, merkez 1 ve toplam 409 değerlerini yalnız uyarı hedefi olarak
  kullanan; hedefe ulaşmak için kayıt taşımayan doğrulama
- Genel merkeze en yakın çember için onay bekleyen merkez önerisi
- K1–K6 spiral uzaklıklarını karşılaştıran, alan kilitlerini gösteren ve
  “en olası dört” filtresi sunan sınır adayları
- 409 kaydın tamamı için onaylı Ana Zinciri, zincir yoksa mevcut spiral kol
  modelini kullanan `inner` / `outer` / `centerline` / `unknown` yerel taraf
  önerileri; kol/tip filtresi, eksen uzaklığı, radyal fark, yerel teğet,
  güven eşiği ve tek-adımlı toplu onay
- Her kol için merkezden dışa tıklanan, ileri/geri düğmeleriyle düzenlenebilen
  ve yalnız 13 seçimle onaylanan **Ana Zincir** taslağı; açık taslağın çalışma
  JSON’unda saklanıp yeniden yüklenmesi
- Onaylı ana zincirlerden en yakın/ikinci en yakın parent, uzaklık oranı,
  normalize uzaklık, yerel açı ve `inner` / `outer` / `centerline` önerisi
- Ana çember üzerinde bağlı yan çember sayısı rozeti ve 13 satırlı yan çember
  durum tablosu; 6. ve 7. konum vurgusu
- Merkez/yay çapı benzerliği ve örtüşme oranıyla mükerrer aday denetimi;
  birleştirmede kilitli veya manuel kayıt önceliği
- Tür, kol, parent, mükerrerlik, güven ve manuel kaynak filtreleri; varsayılan
  olarak yalnız seçilen dairenin kimliğini gösteren etiket modu
- Gerçek alanlardan ayrı `suggested_*` değerleri, alan-bazlı kilitler,
  zaman damgalı düzenleme günlüğü ve tek-adımlı toplu geri alma
- Geometriyi veya sınıflandırmayı değiştirmeden bütün kayıt/alan kilitlerini
  tek onayla kaldıran ve yine tek adımda geri alınabilen **Tüm kilitleri aç**
  komutu

Kol denetimi hiçbir öneriyi kendiliğinden onaylamaz. Merkez, kol, zincir,
parent ve taraf değişiklikleri ancak ilgili arayüz düğmesine açıkça
basıldığında gerçek sınıflandırma alanlarına yazılır.

## Kurulum

Python 3.10 veya daha yenisi gerekir.

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"
```

Sanal ortamı kullanılan kabuğa uygun betikle etkinleştirin:

```bash
# Bash veya Zsh
source .venv/bin/activate
```

```fish
# Fish
source .venv/bin/activate.fish
```

Fish içinde `.venv/bin/activate` kullanılırsa `case builtin not inside of
switch` hatası alınır; Fish için `.venv/bin/activate.fish` gereklidir.

Uygulamayı etkinleştirilmiş ortamda başlatın:

```bash
python app.py
```

Ya da etkinleştirme yapmadan doğrudan sanal ortam Python’unu kullanın:

```bash
.venv/bin/python app.py
```

`ModuleNotFoundError: No module named 'cv2'` görülürse uygulama sistem
Python’uyla çalıştırılmış veya bağımlılıklar kurulmamıştır. Şunu çalıştırın:

```bash
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python app.py
```

Ardından [http://127.0.0.1:5000](http://127.0.0.1:5000) adresini açın.
Sunucu yalnızca `127.0.0.1` üzerinde dinler; hesap veya uzak servis kullanmaz.

## Önerilen çalışma akışı

1. Bir görüntü yükleyin veya gelişmiş sentetik ön işleme örneğini açın.
2. **Ön İşleme** sekmesinde analiz çokgenini belirleyin.
3. İstenmeyen bölgeleri maskeyle boyayın; gerekirse silgi ve geri almayı
   kullanın.
4. Perspektifi dört noktayla tanımlayın veya tarla izi önerisini inceleyin.
   Gerekirse beklenen aralığı girin; izleri gövdelerinden/iki uçlarından
   düzeltip yanlış olanları silin veya yenisini çizin. Otomatik ve manuel
   öneriler açık onay verilmeden uygulanmaz. **Tam kaynak görüntüyü koru**
   açıkken dönüşüm hiçbir kaynak köşesini kırpmaz; geometrik dönüşümden doğan
   siyah köşeler analiz maskesinin dışında kalır.
5. **Perspektif Kalitesi** panelinde öneriyi önce ham tramline ve warp
   ölçümleriyle değerlendirin. Dört nokta seçiminde görüntü köşeleri yerine
   zemindeki aynı düzlemden anlamlı dört referans kullanın.
6. Aydınlatma düzeltmesi ve CLAHE’yi önce/sonra görünümünde ayarlayın.
7. Dört arazi sınıfından temsilî renk örnekleri boyayın.
8. Tarla izi maskesini, üç ortak yan-daire örnek yarıçapını ve bunlardan
   bağımsız ana kol min–maks yarıçap aralığını doğrulayın.
9. İşlem yığınını uygulayın; gerekirse profili kaydedin.
10. **Daireler ve Ölçüm** sekmesine geçip gelişmiş algılamayı çalıştırın.
11. Adayları elle düzeltin, onaylananları kilitleyin ve yeniden algılama
    gerekiyorsa çalıştırın.
12. Genel merkezi seçin. Spiral kollu görüntülerde **Spiral uyumunu öner**
    sonucunu inceleyip uygun bulursanız uygulayın.
13. Altı kolu/simetriyi hesaplayın; **Çalışmayı kaydet** ile düzenlenebilir
    `.work.json` dosyasını kaydedin.
14. **Kol Denetimi** bölümünde raporu yenileyin, merkez önerisini inceleyin,
    gerekirse K1–K6 sınır adaylarını onaylayın ve her kolun 13’lü Ana
    Zincirini oluşturun. **Yerel taraf otomatik önerileri** tablosunu
    yenileyip kaynak yöntemini ve düşük güvenli satırları denetleyin; seçili
    öneriler onaylanırken eski `left/right` değeri `legacy_side` içinde kalır.
15. Yerel taraf ve yan çember parent önerilerini tek tek veya seçili grup olarak
    onaylayın; 68/13/55 özetini ve mükerrer adayları denetleyin.
16. Perspektif kalite hesabını daire ve yapısal kayıtlarla yeniden çalıştırın;
    önce/sonra bileşenlerini ve güven düzeyini denetleyin.
17. PNG/JSON çıktılarıyla birlikte yapısal ve perspektif kalite raporlarını
    dışa aktarın.

Bir çalışmayı geri açmak için önce aynı kaynak görüntüyü açın, sonra
**Çalışma JSON'u yükle** ile `.work.json` veya daha eski sonuç JSON'unu seçin.
Eski sonuç JSON'ları daire ayrıntılarını geri getirir; 1.4 öncesi dosyalarda
silinen otomatik adaylar ve alan-bazlı manuel kilit günlüğü bulunmayabilir.

Fare tekerleği yakınlaştırır. Orta tuş veya `Boşluk + sürükleme` görüntüyü
kaydırır. Etkin araçlar ve kısayollar arayüz araç çubuğunda gösterilir.

## Koordinatlar ve veri şeması

Tüm çizilmiş ön işleme geometrileri kaynak görüntü koordinatlarında saklanır.
Perspektif homografisi ve tersi profil/JSON içinde bulunur. Böylece bir daire
hem işlenmiş görüntüde hem de değişmez kaynak görüntüde izlenebilir.

Her daire şu temel alanları taşır:

| Alan | Anlam |
|---|---|
| `id` | Benzersiz daire kimliği |
| `x`, `y` | Geriye uyumlu işlenmiş merkez koordinatı |
| `processed_x`, `processed_y` | İşlenmiş görüntü merkezi, piksel |
| `source_x`, `source_y` | Özgün kaynak görüntü merkezi, piksel |
| `radius` | İşlenmiş görüntü yarıçapı, piksel |
| `source_radius` | Kaynak koordinat uzayındaki eşdeğer yarıçap |
| `corrected_x`, `corrected_y` | Perspektif sonrası merkez koordinatı |
| `corrected_radius` | Perspektif sonrası eşdeğer alan yarıçapı |
| `source_major_axis`, `source_minor_axis` | Kaynak görünümde fit edilen elips eksenleri |
| `corrected_major_axis`, `corrected_minor_axis` | Düzeltilmiş görünümde fit edilen elips eksenleri |
| `confidence` | `final_confidence` ile uyumlu genel güven |
| `circle_type` | `center`, `main`, `satellite`, `unclassified` veya eski `side` |
| `parent_id` | Yan çemberin bağlı olduğu daire |
| `arm_id` | 1–6 kol kimliği |
| `sequence_index` | Kolda merkezden dışarı sıra |
| `side` | `inner`, `outer`, `centerline`, `unknown` veya merkez için `center` |
| `legacy_side` | Geriye uyumluluk için eski `left` / `right` değeri |
| `local_angle` | Yan çemberin parent ana zincir teğetine göre yerel açısı |
| `normalized_parent_distance` | Parent uzaklığının ana yarıçapa oranı |
| `classification_source` | `legacy`, otomatik-onaylı veya kullanıcı-onaylı kaynak |
| `classification_confidence` | Onaylı sınıflandırma güveni, 0–1 |
| `locked_fields` | Otomatik işlemlerin değiştiremeyeceği yapısal alanlar |
| `suggested_*` | Gerçek değerden ayrı, henüz onaylanmamış öneri alanları |
| `center_distance` | Genel merkeze uzaklık, piksel |
| `center_angle` | Genel merkeze göre açı, derece |
| `edge_score` | Çember çevresindeki kenar desteği |
| `color_score` | Ön plan olasılığı desteği |
| `circularity_score` | Dairesel/elliptik şekil desteği |
| `detector_agreement` | Birleşen algılayıcıların uzlaşması |
| `symmetry_support` | Yerel altılı dönel destek |
| `final_confidence` | Birleştirilmiş güven, 0–1 |
| `stroke_color` | İsteğe bağlı `#RRGGBB` çember çizgi rengi |
| `stroke_width` | İsteğe bağlı çember çizgi kalınlığı, piksel |
| `fill_enabled` | Çember içi dolgunun etkin olup olmadığı |
| `fill_color` | İsteğe bağlı `#RRGGBB` dolgu rengi |
| `fill_opacity` | Dolgu opaklığı, 0–1 |
| `size_class` | Yan standart adayı için `small`, `medium`, `large`; değişken ana kol adayı için `null` |
| `detector` | Adayı üreten/birleştiren yöntemler; değişken ana kol yöntemleri `main_` öneki taşır |
| `locked` | Yeniden algılamada korunup korunmayacağı |
| `source` | `automatic`, `manual` veya `manual_override` |

JSON ayrıca koordinat dönüşüm matrislerini, ön işleme profilini, genel merkezi,
kol fazını, `arm_model`, `arm_twist_degrees_per_100px`, manuel alan kilitlerini,
silinen/bastırılan otomatik aday günlüğünü ve görüntü/geometri simetri
tablolarını saklar. Yapısal çalışma kayıtları ayrıca `structure_state` içindeki
onaylı Ana Zincir kimliklerini ve `edit_history` içindeki önceki/yeni değer,
aktör, zaman ve işlem türünü korur. Fiziksel ölçek sağlanmadıkça bütün ölçüler
piksel cinsindedir.

Yapısal dışa aktarım klasörü şu dosyaları içerir:

- `structure_summary.json`
- `circles_classified.csv`
- `arm_summary.csv`
- `main_sequence.csv`
- `satellite_relationships.csv`
- `boundary_candidates.csv`

Perspektif kalite dışa aktarım klasörü şu dosyaları içerir:

- `perspective_quality_report.json`
- `perspective_circle_metrics.csv`
- `perspective_arm_metrics.csv`
- `tramline_metrics.csv`
- `perspective_before_after.png`
- `roundness_overlay.png`
- `tramline_overlay_corrected.png`
- `arm_scale_heatmap.png`

## Simetri değerleri

`normalized_mae` ve `normalized_rmse`, seçilen merkez çevresindeki ortak
dairesel görüntü alanında `[0, 1]` parlaklık ölçeğinde hesaplanır; küçük değer
daha iyi eşleşmedir. `correlation` için 1 daha iyi eşleşmedir. Geometrik
`normalized_position_error`, döndürülmüş merkezlerin bire bir eşleme hatasını
medyan yarıçapa böler. Bu değerler görüntü/anotasyon kalitesini ölçer; oluşum
nedeni hakkında iddia değildir.

## Sentetik veri ve testler

Altı kollu temel örnek:

```bash
.venv/bin/python -m src.synthetic \
  --output data/raw/synthetic_six_arm.png
```

Küçük, orta ve büyük daireler; temas eden bölgeler; eğik perspektif; uzun
paralel tarla izleri ve değişken aydınlatma içeren ön işleme örneği:

```bash
.venv/bin/python -m src.synthetic \
  --preprocessing \
  --output data/raw/synthetic_preprocessing.png
```

Tüm testleri çalıştırın:

```bash
.venv/bin/python -m pytest
```

Testler ön işleme yığınını, profil serileştirmeyi, maske boyama/silmeyi,
perspektif dönüşümlerini, renk olasılığını, tarla izi önerisini, üç boyut
algılayıcısını, watershed’i, kilitli adayları, çift koordinatları, simetriyi ve
Flask API/dışa aktarım akışını kapsar. Perspektif kalite testleri ayrıca
tramline iyileşmesini, elips/kol/K5–K6 metriklerini, warp cezasını, rapor
dosyalarını ve anotasyon girdisinin değişmeden kalmasını doğrular.

## Proje yapısı

```text
crop-geometry-lab/
├── data/
│   ├── raw/
│   ├── corrected/
│   └── annotations/
├── hypotheses/
│   └── milk_hill.yaml
├── src/
│   ├── preprocessing.py
│   ├── perspective.py
│   ├── perspective_quality.py
│   ├── multiscale_detector.py
│   ├── circle_detector.py
│   ├── manual_editor.py
│   ├── arm_classifier.py
│   ├── feature_table.py
│   ├── symmetry_tests.py
│   ├── visualizer.py
│   ├── synthetic.py
│   ├── torus_models.py
│   └── webapp.py
├── tests/
├── web/
├── outputs/
├── app.py
└── README.md
```

`src/torus_models.py` bilinçli olarak yalnızca hata veren bir yer tutucudur.
