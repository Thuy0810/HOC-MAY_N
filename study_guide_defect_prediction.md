# Tài Liệu Học Tập: Dự Đoán Số Lượng Lỗi Phần Mềm Bằng Deep Learning

Tài liệu này được biên soạn nhằm giúp bạn hiểu sâu sắc và làm chủ toàn bộ kiến thức, thuật toán, và cấu trúc kỹ thuật của case study **Dự đoán số lượng lỗi phần mềm sử dụng mạng nơ-ron tích chập (CNN) và mạng nơ-ron đa tầng (MLP)** kết hợp kỹ thuật cân bằng dữ liệu **SMOTEND**. 

Tài liệu này bám sát bài báo nghiên cứu gốc: *Predicting the Number of Software Faults using Deep Learning (ETASR 2024)*.

---

## MỤC LỤC
1. [Khái Quát Bài Toán & Ý Nghĩa Thực Tiễn](#1-khái-quát-bài-toán--ý-nghĩa-thực-tiễn)
2. [Chi Tiết Quy Trình Thực Nghiệm (Pipeline)](#2-chi-tiết-quy-trình-thực-nghiệm-pipeline)
3. [Tìm Hiểu Về Các Thuộc Tính Phần Mềm (Software Metrics)](#3-tìm-hiểu-về-các-thuộc-tính-phần-mềm-software-metrics)
4. [Các Bước Tiền Xử Lý Dữ Liệu Quan Trọng](#4-các-bước-tiền-xử-lý-dữ-liệu-quan-trọng)
5. [Thuật Toán SMOTEND Cho Bài Toán Hồi Quy](#5-thuật-toán-smotend-cho-bài-toán-hồi-quy)
6. [Kiến Trúc Mô Hình Học Sâu Song Song (Multi-Input CNN & MLP)](#6-kiến-trúc-mô-hình-học-sâu-song-song-multi-input-cnn--mlp)
7. [Các Phương Pháp Đánh Giá & Baseline](#7-các-phương-pháp-đánh-giá--baseline)
8. [Phân Tích Kết Quả & Hướng Dẫn Viết Báo Cáo](#8-phân-tích-kết-quả--hướng-dẫn-viết-báo-cáo)

---

## 1. Khái Quát Bài Toán & Ý Nghĩa Thực Tiễn

### Phân biệt Phân loại (Classification) và Hồi quy (Regression)
* **SFP truyền thống (Classification)**: Dự đoán một class/module có lỗi (`1`) hay không có lỗi (`0`).
* **Bài toán trong bài báo (Regression)**: Dự đoán chính xác **số lượng lỗi** (ví dụ: 0, 1, 5, 20 lỗi) trong từng class.

### Tại sao dự đoán số lượng lỗi lại quan trọng hơn?
Trong thực tế phát triển phần mềm, nguồn lực kiểm thử (tester, thời gian, chi phí máy móc) luôn có hạn. Nếu mô hình chỉ trả về "Class A có lỗi" và "Class B có lỗi", tester sẽ không biết nên tập trung vào đâu trước. 
Khi biết **Class A dự kiến có 15 lỗi** và **Class B dự kiến có 1 lỗi**, tester sẽ ngay lập tức ưu tiên kiểm thử kỹ lưỡng Class A. Điều này giúp tối ưu hóa chi phí phát hiện lỗi và tăng chất lượng sản phẩm nhanh nhất.

---

## 2. Chi Tiết Quy Trình Thực Nghiệm (Pipeline)

Quy trình thực nghiệm trong notebook tuân theo sơ đồ khoa học sau:

```
 Dữ liệu thô (Lucene, Poi, Xalan)
       │
       ▼
 Làm sạch dữ liệu (Xóa NaN, trùng lặp, giữ 20 metrics)
       │
       ▼
 Chia tập dữ liệu (70% Train, 30% Test)
       │
 ┌─────┴────────────────────────────────────────┐
 │                                              │
 ▼                                              ▼
Thí nghiệm 1: Không dùng SMOTEND          Thí nghiệm 2: Có dùng SMOTEND-style
(Dữ liệu mất cân bằng)                     (Dữ liệu cân bằng giữa bug=0 và bug>0)
 │                                              │
 ├──────────────────────────────────────────────┤
 ▼
Tiền xử lý (Log-transform Target & Standardize Features)
       │
       ▼
Chia dữ liệu thành 5 nhóm đầu vào (5 Groups of 4 metrics)
       │
       ├──────────────────────────────────────────────┐
       ▼                                              ▼
Huấn luyện Deep Learning (CNN & MLP)          Huấn luyện Baseline (SVR & DTR)
       │                                              │
       └──────────────────────┬───────────────────────┘
                              ▼
                  Đánh giá & So sánh kết quả
                  (Độ đo MSE & Hệ số Kendall)
```

---

## 3. Tìm Hiểu Về Các Thuộc Tính Phần Mềm (Software Metrics)

Bộ dữ liệu PROMISE trích xuất các thuộc tính hướng đối tượng (OO metrics). Trong đó, 20 thuộc tính được sử dụng trong mô hình được chia thành **5 nhóm độc lập** tương ứng với các chiều đo chất lượng phần mềm:

| Nhóm | Tên Metric | Ý Nghĩa / Định Nghĩa |
| :--- | :--- | :--- |
| **Nhóm 1** | **wmc** | Weighted Methods per Class: Tổng độ phức tạp của các phương thức trong class. |
| (Complexity | **dit** | Depth of Inheritance Tree: Độ sâu tối đa của class trong cây kế thừa. |
| & Inheritance) | **noc** | Number of Children: Số lượng class con kế thừa trực tiếp từ class này. |
| | **cbo** | Coupling Between Object classes: Số lượng class khác liên kết với class này. |
| **Nhóm 2** | **rfc** | Response for a Class: Số lượng phương thức có thể được gọi từ class này. |
| (Response & | **lcom** | Lack of Cohesion in Methods: Độ thiếu gắn kết giữa các phương thức. |
| Cohesion) | **ca** | Afferent Couplings: Số lượng class ngoài phụ thuộc vào class hiện tại. |
| | **ce** | Efferent Couplings: Số lượng class ngoài mà class hiện tại phụ thuộc vào. |
| **Nhóm 3** | **npm** | Number of Public Methods: Số lượng phương thức public trong class. |
| (Size & | **lcom3** | Trạng thái thiếu gắn kết phiên bản cải tiến (thang đo từ 0 đến 2). |
| Encapsulation) | **loc** | Lines of Code: Số dòng code của class. |
| | **dam** | Data Access Metric: Tỷ lệ thuộc tính private/protected trên tổng số thuộc tính. |
| **Nhóm 4** | **moa** | Measure of Aggregation: Số lượng thuộc tính có kiểu dữ liệu là class khác. |
| (Aggregation & | **mfa** | Measure of Functional Abstraction: Tỷ lệ phương thức kế thừa từ class cha. |
| Abstraction) | **cam** | Cohesion Among Methods: Độ gắn kết giữa các tham số của phương thức. |
| | **ic** | Inheritance Coupling: Số lượng class cha liên kết qua kế thừa. |
| **Nhóm 5** | **cbm** | Coupling Between Methods: Số lượng liên kết giữa các phương thức. |
| (Advanced | **amc** | Average Method Size: Kích thước trung bình của các phương thức trong class. |
| Complexity) | **max_cc** | Maximum Cyclomatic Complexity: Độ phức tạp tuần hoàn lớn nhất trong class. |
| | **avg_cc** | Average Cyclomatic Complexity: Độ phức tạp tuần hoàn trung bình. |

---

## 4. Các Bước Tiền Xử Lý Dữ Liệu Quan Trọng

### 4.1. Khắc phục lệch phải cực đoan bằng Log-transform (`log1p`)
Trong thực tế, số lượng lỗi phân bố theo quy luật Pareto (80% lỗi nằm trong 20% class). Điều này có nghĩa là đa số class có 0 lỗi, một số ít class có 1-2 lỗi và cực kỳ hiếm các class có rất nhiều lỗi (như 30, 47 lỗi). 

Nếu huấn luyện mô hình trực tiếp trên số lượng lỗi thô này, hàm loss (MSE) sẽ bị ảnh hưởng cực kỳ lớn bởi các class có số lỗi khổng lồ (bình phương sai số cực lớn), làm lu mờ hành vi của các class có ít lỗi hơn.
Giải pháp là áp dụng **Log1p transformation**:
$$y_{log} = \ln(y_{original} + 1)$$

* **Tại sao dùng $\ln(y+1)$ thay vì $\ln(y)$?** Vì nếu $y=0$ (class không lỗi), $\ln(0)$ sẽ không xác định ($-\infty$). Phép cộng thêm $1$ giúp đảm bảo đầu ra luôn xác định: $\ln(0+1) = 0$.
* Khi đánh giá kết quả dự đoán thô, ta khôi phục về thang đo gốc bằng hàm ngược:
$$y_{original} = e^{y_{log}} - 1$$

### 4.2. Chuẩn hóa đặc trưng (Standardization)
Các metric có đơn vị rất khác nhau (ví dụ: `dit` chỉ từ 1-5, trong khi `loc` lên tới hàng ngàn dòng). Ta dùng `StandardScaler`:
$$x_{standardized} = \frac{x - \mu}{\sigma}$$
Đưa tất cả về phân bố có trung bình bằng 0 và độ lệch chuẩn bằng 1, giúp thuật toán Gradient Descent hội tụ nhanh hơn.

---

## 5. Thuật Toán SMOTEND Cho Bài Toán Hồi Quy

**SMOTE (Synthetic Minority Over-sampling Technique)** thường dùng cho phân loại nhị phân để sinh thêm mẫu thiểu số bằng cách nội suy. 
Trong bài toán hồi quy lỗi phần mềm, ta định nghĩa:
* **Majority (Nhóm đa số)**: Các mẫu không có lỗi (`bug = 0`).
* **Minority (Nhóm thiểu số)**: Các mẫu có lỗi (`bug > 0`).

### Các bước thuật toán SMOTEND-style trong notebook:
1. Xác định tập các class thiểu số (faulty classes).
2. Với mỗi mẫu có lỗi $x_i$, tính khoảng cách Minkowski đến các mẫu có lỗi khác:
   $$D(a, b) = \left( \sum_{j=1}^{d} |a_j - b_j|^r \right)^{1/r}$$
   (Trong code chọn $r=2.0$, tức khoảng cách Euclidean).
3. Chọn ngẫu nhiên một trong $k$ láng giềng gần nhất $x_{knn}$.
4. Sinh mẫu mới $x_{new}$ bằng cách nội suy tuyến tính:
   $$x_{new} = x_i + \lambda \times (x_{knn} - x_i)$$
   với $\lambda \in [0, 1]$ ngẫu nhiên.
5. **Nội suy số lượng lỗi (Target `bug`)**: 
   Số lượng lỗi của mẫu mới được tính toán dựa trên trọng số khoảng cách đến $x_i$ và $x_{knn}$, giúp số lượng lỗi của mẫu mới tự nhiên và bám sát thực tế nhất.

---

## 6. Kiến Trúc Mô Hình Học Sâu Song Song (Multi-Input CNN & MLP)

Điểm cải tiến độc đáo của bài báo là thiết kế **kiến trúc song song** tương ứng với 5 nhóm thuộc tính chất lượng mã nguồn thay vì đưa trực tiếp 20 thuộc tính vào một mạng học sâu phẳng.

### 6.1. Mô hình Multi-Input CNN
* **Đầu vào**: 5 nhánh song song, mỗi nhánh nhận 1 Tensor đầu vào kích thước `(4, 1)`.
* **Trích xuất đặc trưng**: Mỗi nhánh đi qua một lớp tích chập 1 chiều **Conv1D** (filters=32, kernel_size=2). Lớp này học mối quan hệ tương quan cục bộ giữa 4 thuộc tính trong cùng một nhóm.
* **Giảm chiều**: Lớp **MaxPooling1D** (pool_size=2) thu gọn đặc trưng nổi bật nhất.
* **Chống quá khớp**: Lớp **Dropout** (rate=0.2) ngắt kết nối ngẫu nhiên 20% nơ-ron trong quá trình huấn luyện để mô hình không phụ thuộc quá mức vào bất kỳ đặc trưng nào.
* **Ghép nối (Merge)**: Lớp **Flatten** đưa đầu ra mỗi nhánh về dạng vector 1D, sau đó lớp **Concatenate** gộp 5 vector này thành 1 vector duy nhất chứa thông tin trích xuất của cả 5 nhóm.
* **Dự đoán**: Đi qua một mạng Dense kết nối đầy đủ (64 units) và đưa ra dự báo hồi quy tuyến tính ở lớp cuối cùng (1 unit).

#### Triển khai code Keras hoàn chỉnh:
```python
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Concatenate, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam

def build_cnn_model(learning_rate=0.001):
    # Định nghĩa 5 nhánh đầu vào (mỗi nhóm gồm 4 metrics)
    inputs = [Input(shape=(4, 1), name=f"input_group_{i+1}") for i in range(5)]
    
    flattened_outputs = []
    for i, inp in enumerate(inputs):
        # Lớp Convolutional trích xuất đặc trưng cục bộ
        x = Conv1D(filters=32, kernel_size=2, activation='relu', 
                   kernel_initializer='glorot_uniform', padding='same')(inp)
        # Giảm chiều đặc trưng
        x = MaxPooling1D(pool_size=2)(x)
        # Tránh overfitting
        x = Dropout(0.2)(x)
        # Làm phẳng để chuẩn bị ghép nối
        x = Flatten()(x)
        flattened_outputs.append(x)
        
    # Lớp ghép nối (Merge Layer)
    merged = Concatenate()(flattened_outputs)
    
    # Lớp kết nối đầy đủ (Fully Connected)
    x = Dense(64, activation='relu', kernel_initializer='glorot_uniform')(merged)
    x = Dropout(0.2)(x)
    
    # Lớp đầu ra hồi quy tuyến tính
    output = Dense(1, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=output, name="Parallel_CNN")
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model
```

---

### 6.2. Mô hình Multi-Input MLP
Mô hình MLP song song có cấu trúc tương tự CNN nhưng đơn giản hơn: Thay vì dùng Conv1D và MaxPooling, mỗi nhánh đầu vào dạng phẳng kích thước `(4,)` sẽ trực tiếp đi qua một lớp **Dense** (16 units) để học biểu diễn phi tuyến tính của riêng nhóm đó.

#### Triển khai code Keras hoàn chỉnh:
```python
def build_mlp_model(learning_rate=0.001):
    # Định nghĩa 5 nhánh đầu vào
    inputs = [Input(shape=(4,), name=f"input_group_{i+1}") for i in range(5)]
    
    dense_outputs = []
    for i, inp in enumerate(inputs):
        # Trực tiếp đi qua lớp Dense ẩn
        x = Dense(16, activation='relu', kernel_initializer='glorot_uniform')(inp)
        x = Dropout(0.2)(x)
        dense_outputs.append(x)
        
    # Ghép nối các nhánh
    merged = Concatenate()(dense_outputs)
    
    # Lớp kết nối đầy đủ phía sau
    x = Dense(64, activation='relu', kernel_initializer='glorot_uniform')(merged)
    x = Dropout(0.2)(x)
    
    # Lớp đầu ra hồi quy
    output = Dense(1, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=output, name="Parallel_MLP")
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model
```

---

## 7. Các Phương Pháp Đánh Giá & Baseline

### 7.1. Hệ số tương quan thứ bậc Kendall (Kendall's $\tau$)
Là độ đo phi tham số đo lường mức độ tương quan về mặt **thứ tự** (rank) giữa giá trị thực tế ($y$) và dự đoán ($\hat{y}$):
$$\tau = \frac{C - D}{\frac{1}{2} n(n-1)}$$
Trong đó:
* $C$: Số cặp đồng hướng (concordant pairs) - cả thực tế và dự đoán đều tăng.
* $D$: Số cặp nghịch hướng (discordant pairs) - một bên tăng một bên giảm.
* Giá trị $\tau$ chạy từ $-1$ (nghịch tương quan hoàn toàn) đến $+1$ (tương quan thứ bậc hoàn hảo). Hệ số Kendall càng cao chứng tỏ khả năng sắp xếp thứ tự ưu tiên kiểm thử của mô hình càng chính xác.

### 7.2. Mean Squared Error (MSE)
Sai số bình phương trung bình giữa giá trị log thực tế và giá trị log dự báo:
$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_{log\_real} - y_{log\_pred})^2$$

### 7.3. Baseline Machine Learning
* **Decision Tree Regressor (DTR)**: Cực kỳ trực quan, chia nhánh dữ liệu theo các ngưỡng tối ưu của đặc trưng. Hoạt động như một baseline nhanh.
* **Support Vector Regressor (SVR)**: Tìm một siêu phẳng trong không gian nhiều chiều sao cho sai lệch dự báo nằm trong một biên sai số $\epsilon$ cho trước, sử dụng Kernel phi tuyến RBF.

---

## 8. Phân Tích Kết Quả & Hướng Dẫn Viết Báo Cáo

Dưới đây là các kết quả khoa học cực kỳ quan trọng thu được từ bài thực nghiệm mà bạn cần đưa vào và phân tích trong báo cáo khoa học của mình:

### 8.1. Bảng kết quả thực nghiệm chuẩn từ bài báo
Bạn nên đưa bảng so sánh này vào chương **Kết quả và thảo luận**:

| Thí nghiệm | Mô hình | Tập dữ liệu | Hệ số Kendall (Cao là tốt) | Chỉ số MSE (Thấp là tốt) |
| :--- | :--- | :--- | :---: | :---: |
| **Thí nghiệm 1** | **CNN** | Train / Test | 0.190 / 0.162 | 1.776 / 1.316 |
| *(Chưa qua SMOTEND)* | **MLP** | Train / Test | 0.186 / 0.183 | 1.887 / 1.730 |
| **Thí nghiệm 2** | **CNN** | Train / Test | 0.361 / 0.363 | 0.222 / 0.218 |
| *(Đã cân bằng SMOTEND)*| **MLP** | Train / Test | **0.444 / 0.416** | **0.185 / 0.195** |

### 8.2. Nhận xét phân tích kết quả chuyên sâu (Dành cho báo cáo)
1. **Tác động mạnh mẽ của SMOTEND**: 
   * Trước khi áp dụng SMOTEND (Thí nghiệm 1), cả hai mô hình học sâu hoạt động khá kém. Hệ số tương quan thứ bậc Kendall chỉ đạt dưới 0.19 và chỉ số MSE rất cao (trên 1.3). Lý do là mô hình bị ảnh hưởng nặng nề bởi các mẫu đa số (`bug = 0`), khiến nó có xu hướng dự đoán số lỗi thiên về 0 cho tất cả các class.
   * Sau khi áp dụng SMOTEND (Thí nghiệm 2), hiệu suất tăng vọt đáng ngạc nhiên. Hệ số Kendall tăng gấp đôi (đạt tới 0.416 ở MLP) và sai số MSE giảm sâu (dưới 0.22). Điều này chứng minh việc giải quyết bài toán mất cân bằng dữ liệu bằng cách sinh dữ liệu tổng hợp dựa trên láng giềng gần nhất là cực kỳ hiệu quả đối với dự đoán lỗi phần mềm.
2. **Sự vượt trội của MLP so với CNN trên dữ liệu số**:
   * Ngược lại với các nghiên cứu xử lý ảnh hoặc văn bản nơi CNN vượt trội, trong case study này trên tập kiểm thử đã cân bằng, **MLP cho kết quả tốt hơn hẳn CNN** (Kendall 0.416 so với 0.363; MSE 0.195 so với 0.218).
   * **Giải thích**: Các thuộc tính chất lượng phần mềm là dữ liệu dạng số thuần túy (numerical tabular data) không có cấu trúc không gian liên tục hay tính tuần tự cao như hình ảnh hoặc âm thanh. Cấu trúc CNN phức tạp với các bộ lọc tích chập cục bộ đôi khi làm phức tạp hóa mối quan hệ của các biến số độc lập, trong khi mạng MLP với khả năng biểu diễn trực tiếp và phi tuyến tính mạnh mẽ dễ dàng tìm thấy các ánh xạ tối ưu từ thuộc tính phần mềm sang số lượng lỗi thực tế.
