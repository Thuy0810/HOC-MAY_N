# Báo Cáo Chi Tiết và Phân Tích Bộ Dữ Liệu PROMISE (Software Defect Prediction)

Bộ dữ liệu **PROMISE Software Defect Prediction** là một trong những bộ dữ liệu chuẩn (benchmark datasets) phổ biến nhất được sử dụng trong nghiên cứu kỹ thuật phần mềm, đặc biệt là bài toán **Dự đoán lỗi phần mềm (Software Fault/Defect Prediction - SFP)**. 

Tài liệu này cung cấp một cái nhìn toàn diện và chi tiết về cấu trúc, ý nghĩa các thuộc tính, phân bố thống kê, và các thách thức đặc thù của bộ dữ liệu PROMISE (gồm 3 dự án mã nguồn mở Java nổi tiếng: `lucene`, `poi`, và `xalan`) được sử dụng trong thực nghiệm.

---

## 1. Tổng Quan Về Các Dự Án & Phiên Bản Sử Dụng

Thực nghiệm sử dụng 3 dự án lớn từ Apache với các phiên bản cụ thể nhằm tái lập thực nghiệm khoa học chuẩn xác:

| Dự án | Phiên bản | Mô tả dự án |
| :--- | :--- | :--- |
| **Lucene** | 2.0, 2.2, 2.4 | Thư viện tìm kiếm và phân tích cú pháp văn bản hiệu năng cao bằng Java. |
| **POI** | 1.5, 2.0, 2.5, 3.0 | API Java giúp đọc/ghi các định dạng tệp tin của Microsoft Office (Word, Excel, PowerPoint). |
| **Xalan** | 2.4, 2.5, 2.6, 2.7 | Công cụ xử lý và chuyển đổi tài liệu XML sang HTML, text hoặc định dạng XML khác sử dụng XSLT. |

### Bảng Thống Kê Chi Tiết Dữ Liệu Thực Nghiệm

Dữ liệu thô sau khi được làm sạch (loại bỏ các dòng chứa giá trị thiếu `NaN` và các bản ghi trùng lặp) có đặc điểm cấu trúc như sau:

| Dự án | Phiên bản | Số Lớp (Rows) | Lớp Bị Lỗi (Bug > 0) | Lớp Lành Lặn (Bug = 0) | Số Lỗi Max | Số Lỗi Trung Bình | Tỷ Lệ Lỗi (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **lucene** | 2.0 | 193 | 91 | 102 | 22 | 1.39 | 47.15% |
| **lucene** | 2.2 | 209 | 141 | 68 | 47 | 1.97 | 67.46% |
| **lucene** | 2.4 | 312 | 196 | 116 | 30 | 2.00 | 62.82% |
| **poi** | 1.5 | 219 | 130 | 89 | 20 | 1.48 | 59.36% |
| **poi** | 2.0 | 218 | 35 | 183 | 2 | 0.17 | 16.06% |
| **poi** | 2.5 | 269 | 213 | 56 | 11 | 1.64 | 79.18% |
| **poi** | 3.0 | 341 | 250 | 91 | 19 | 1.38 | 73.31% |
| **xalan** | 2.4 | 688 | 110 | 578 | 7 | 0.23 | 15.99% |
| **xalan** | 2.5 | 614 | 364 | 250 | 9 | 0.83 | 59.28% |
| **xalan** | 2.6 | 437 | 268 | 169 | 9 | 1.07 | 61.33% |
| **xalan** | 2.7 | 657 | 656 | 1 | 8 | 1.33 | 99.85% |
| **Tổng cộng**| **-** | **4,157** | **2,454** | **1,703** | **47** | **1.14** | **59.03%** |

---

## 2. Ý Nghĩa Chi Tiết 20 Thuộc Tính Phần Mềm (Software Metrics)

Bộ dữ liệu PROMISE trích xuất các thuộc tính tĩnh của mã nguồn ở cấp độ Lớp (Class level). 20 thuộc tính này (thuộc bộ chỉ số CK Metrics nổi tiếng của Chidamber & Kemerer cùng một số chỉ số bổ sung) được chia thành **5 nhóm** để đưa vào mô hình học sâu song song:

### Nhóm 1: Cấu trúc & Kế thừa (Complexity & Inheritance)
Học các thông tin liên quan đến vị trí và độ phức tạp của lớp trong cây phân cấp kế thừa và liên kết cơ bản.
*   **wmc (Weighted Methods per Class):** Tổng độ phức tạp của tất cả các phương thức trong một lớp. Nếu độ phức tạp của mỗi phương thức được tính bằng độ phức tạp tuần hoàn (Cyclomatic Complexity) bằng 1, thì WMC chính là số lượng phương thức của lớp đó.
*   **dit (Depth of Inheritance Tree):** Độ sâu tối đa của lớp trong cây kế thừa từ lớp gốc (Object). Lớp có `dit` càng lớn thì càng kế thừa nhiều hành vi từ lớp cha, làm tăng tính phức tạp khi kiểm thử và gỡ lỗi.
*   **noc (Number of Children):** Số lượng lớp con trực tiếp kế thừa từ lớp này. Một lớp có nhiều con đòi hỏi phải thiết kế và kiểm thử kỹ lưỡng hơn vì bất kỳ lỗi nào ở lớp cha cũng sẽ lan truyền xuống các lớp con.
*   **cbo (Coupling Between Object classes):** Số lượng các lớp khác mà lớp này liên kết tới. Sự liên kết xảy ra qua việc gọi phương thức, truy cập thuộc tính, kế thừa hoặc định nghĩa tham số. `cbo` cao thể hiện tính đóng gói kém, làm mã nguồn khó bảo trì.

### Nhóm 2: Phản hồi & Gắn kết (Response & Cohesion)
Đo lường sự tương tác của lớp với môi trường xung quanh và độ tập trung chức năng bên trong lớp.
*   **rfc (Response for a Class):** Kích thước của tập phản hồi của lớp. Đây là tổng số phương thức có thể được thực thi trực tiếp để phản hồi lại một thông điệp gửi tới đối tượng của lớp đó (bao gồm các phương thức nội bộ và các phương thức của lớp khác được gọi từ bên trong lớp này).
*   **lcom (Lack of Cohesion in Methods):** Thang đo sự thiếu gắn kết giữa các phương thức. Nó đo lường số lượng cặp phương thức trong lớp không chia sẻ chung bất kỳ thuộc tính nào trừ đi số lượng cặp phương thức có chia sẻ thuộc tính. LCOM cao chứng tỏ lớp đang thực hiện quá nhiều chức năng không liên quan và nên được tách nhỏ (Vi phạm Nguyên lý Đơn Nhiệm - Single Responsibility Principle).
*   **ca (Afferent Couplings):** Số lượng lớp bên ngoài phụ thuộc vào lớp này (Độ liên kết hướng vào).
*   **ce (Efferent Couplings):** Số lượng lớp bên ngoài mà lớp này phụ thuộc vào để hoạt động (Độ liên kết hướng ra).

### Nhóm 3: Kích thước & Đóng gói (Size & Encapsulation)
Đo lường quy mô dòng code và mức độ che giấu thông tin của lớp.
*   **npm (Number of Public Methods):** Số lượng phương thức public có thể gọi từ bên ngoài lớp.
*   **lcom3 (Lack of Cohesion in Methods - Version 3):** Chỉ số thiếu gắn kết cải tiến, có giá trị chuẩn hóa chạy từ 0 đến 2. Giá trị gần 0 cho thấy sự gắn kết rất cao; giá trị $\ge 1$ cho thấy sự thiếu gắn kết nghiêm trọng.
*   **loc (Lines of Code):** Số dòng code của lớp. Đây là metric truyền thống đơn giản nhưng cực kỳ hiệu quả để ước lượng kích thước phần mềm.
*   **dam (Data Access Metric):** Tỷ lệ giữa số thuộc tính được đóng gói (private hoặc protected) trên tổng số thuộc tính của lớp. `dam` gần 1 cho thấy lớp tuân thủ tốt nguyên lý đóng gói dữ liệu.

### Nhóm 4: Tích hợp & Trừu tượng (Aggregation & Abstraction)
Học các kiểu dữ liệu phức tạp và mức độ trừu tượng hóa phương thức.
*   **moa (Measure of Aggregation):** Số lượng thuộc tính có kiểu dữ liệu là một lớp tự định nghĩa khác (thay vì các kiểu dữ liệu nguyên bản như int, float, string). Đo lường mức độ tích hợp đối tượng.
*   **mfa (Measure of Functional Abstraction):** Tỷ lệ giữa số phương thức kế thừa từ lớp cha trên tổng số phương thức mà lớp này sở hữu.
*   **cam (Cohesion Among Methods):** Độ gắn kết giữa các phương thức dựa trên sự tương đồng về kiểu dữ liệu của các tham số đầu vào. Giá trị chạy từ 0 đến 1, càng gần 1 càng tốt.
*   **ic (Inheritance Coupling):** Số lượng lớp cha liên kết gián tiếp thông qua kế thừa khi một phương thức kế thừa gọi đến một phương thức khác của lớp cha.

### Nhóm 5: Độ phức tạp nâng cao (Advanced Complexity)
Đo lường cấu trúc chi tiết của các hàm bên trong lớp.
*   **cbm (Coupling Between Methods):** Số lượng liên kết cụ thể giữa các phương thức của lớp cha và lớp con thông qua cơ chế ghi đè (overriding) hoặc gọi trực tiếp hàm cha.
*   **amc (Average Method Size):** Kích thước trung bình của các phương thức (tính bằng số dòng code hoặc mã máy).
*   **max_cc (Maximum Cyclomatic Complexity):** Độ phức tạp tuần hoàn lớn nhất trong số các phương thức của lớp. Đo lường nhánh rẽ tối đa (if/else, vòng lặp) của một phương thức đơn lẻ.
*   **avg_cc (Average Cyclomatic Complexity):** Độ phức tạp tuần hoàn trung bình của các phương thức trong lớp.

---

## 3. Biến Mục Tiêu (`bug`) và Phân Bố Thống Kê

Biến mục tiêu **`bug`** trong bài toán này là một biến số nguyên không âm ($y \ge 0$), đại diện cho **số lượng lỗi được phát hiện và sửa chữa** trong lớp phần mềm đó trong suốt vòng đời phiên bản.

> [!NOTE]
> Khác với bài toán phân loại lỗi nhị phân truyền thống (chỉ dự báo Lớp có lỗi `1` hay không lỗi `0`), bài toán hồi quy số lượng lỗi cung cấp thông tin giá trị hơn rất nhiều cho tester để ưu tiên kiểm thử (prioritization) dựa trên mức độ nghiêm trọng của lỗi.

### Bảng Phân Phối Tần Suất Lỗi Trên Toàn Bộ Tập Dữ Liệu

Dưới đây là bảng phân phối tần suất thực tế của biến mục tiêu `bug` trong tập dữ liệu tổng hợp sau làm sạch:

| Số Lượng Lỗi (Bug) | Số Lớp (Count) | Tỷ Lệ Phần Trăm (%) | Tỷ Lệ Tích Lũy (%) |
| :---: | :---: | :---: | :---: |
| **0** | 1,703 | 40.97% | 40.97% |
| **1** | 1,497 | 36.01% | 76.98% |
| **2** | 579 | 13.93% | 90.91% |
| **3** | 148 | 3.56% | 94.47% |
| **4** | 91 | 2.19% | 96.66% |
| **5** | 49 | 1.18% | 97.84% |
| **6** | 26 | 0.63% | 98.47% |
| **7** | 17 | 0.41% | 98.88% |
| **8** | 8 | 0.19% | 99.07% |
| **9** | 8 | 0.19% | 99.26% |
| **11** | 12 | 0.29% | 99.55% |
| **10, 12-47** | 19 | 0.45% | 100.00% |

### Đặc Điểm Phân Phối Cực Đoan:
1.  **Lệch Phải Nặng (Right-Skewed/Long-Tail Distribution):** Hơn **90%** số lớp chỉ có từ 0 đến 2 lỗi. Tuy nhiên, phần đuôi bên phải kéo rất dài đến tận 47 lỗi. Đây là minh chứng rõ ràng cho **Quy luật Pareto (Quy luật 80/20)** trong công nghệ phần mềm: phần lớn lỗi tập trung ở một số ít các lớp cực kỳ phức tạp.
2.  **Lệch Không (Zero-Inflation):** Nhóm lớp không có lỗi chiếm tỷ lệ lớn nhất (40.97%). Điều này gây khó khăn lớn cho các mô hình hồi quy vì hàm lỗi MSE dễ bị thống trị bởi các lớp không lỗi, khiến mô hình có xu hướng dự đoán số lượng lỗi tiệm cận về 0 cho mọi trường hợp.

---

## 4. Phân Tích Mối Tương Quan Giữa Đặc Trưng và Lỗi

Bảng dưới đây thể hiện hệ số tương quan tuyến tính (Pearson correlation coefficient) giữa từng đặc trưng phần mềm và biến mục tiêu `bug`, được sắp xếp từ cao xuống thấp:

| Metric | Hệ Số Tương Quan với Bug | Nhận Xét & Ý Nghĩa Thực Tế |
| :--- | :---: | :--- |
| **rfc** | **0.3923** | **Mạnh nhất.** Class liên kết gọi càng nhiều phương thức bên ngoài thì càng dễ lỗi. |
| **wmc** | **0.3225** | Số lượng phương thức và độ phức tạp tính toán càng cao, nguy cơ lỗi càng lớn. |
| **loc** | **0.3116** | Số dòng code càng lớn thì xác suất xuất hiện lỗi cú pháp/logic càng cao. |
| **lcom** | 0.2405 | Sự thiếu gắn kết trong lớp dẫn tới cấu trúc lỏng lẻo, dễ phát sinh lỗi khi sửa đổi. |
| **npm** | 0.2381 | Số lượng phương thức public nhiều làm tăng diện tiếp xúc và phụ thuộc của hệ thống. |
| **moa** | 0.2238 | Sự tích hợp đối tượng phức tạp làm tăng độ gắn kết vật lý giữa các lớp. |
| **ce** | 0.2136 | Lớp phụ thuộc vào quá nhiều lớp khác dễ bị lỗi lan truyền khi các lớp kia thay đổi. |
| **cbo** | 0.1565 | Độ liên kết tổng quát giữa các đối tượng. |
| **max_cc** | 0.1226 | Nhánh rẽ điều kiện phức tạp nhất của phương thức đơn lẻ. |
| **dam** | 0.1030 | Tỷ lệ đóng gói thuộc tính. |
| **ca** | 0.0706 | Độ liên kết hướng vào. |
| **avg_cc** | 0.0574 | Độ phức tạp tuần hoàn trung bình. |
| **amc** | 0.0410 | Kích thước trung bình của phương thức. |
| **cbm** | 0.0342 | Liên kết giữa các phương thức cha con. |
| **noc** | 0.0102 | Số lớp con trực tiếp hầu như không tương quan tuyến tính trực tiếp với bug. |
| **ic** | 0.0001 | Liên kết kế thừa gián tiếp không có tương quan tuyến tính. |
| **dit** | -0.0692 | Kế thừa càng sâu thì tương quan âm nhẹ (có thể do các lớp sâu thường ổn định). |
| **lcom3** | -0.0818 | Tương quan âm nhẹ với phiên bản lcom cải tiến. |
| **mfa** | -0.0834 | Tỷ lệ phương thức kế thừa càng cao thì xu hướng lỗi càng giảm nhẹ. |
| **cam** | **-0.1975** | **Tương quan âm mạnh nhất.** Độ gắn kết tham số giữa các phương thức càng cao (lớp được thiết kế mạch lạc, tập trung) thì tỷ lệ lỗi **càng thấp**. |

> [!TIP]
> **Đa cộng tuyến (Multicollinearity):** Các đặc trưng như `loc`, `wmc`, và `rfc` có độ tương quan chéo cực kỳ cao với nhau (thường > 0.8) vì class nhiều dòng code hiển nhiên sẽ có nhiều phương thức và gọi nhiều hàm. Việc sử dụng mạng Neural song song giúp giảm thiểu ảnh hưởng của đa cộng tuyến nhờ khả năng tự động học biểu diễn phi tuyến tính thay vì cộng tuyến phẳng.

---

## 5. Các Bước Tiền Xử Lý Dữ Liệu Quan Trọng

Để huấn luyện các mô hình Machine Learning và Deep Learning đạt hiệu năng tối ưu trên bộ dữ liệu PROMISE, 3 kỹ thuật xử lý dữ liệu sau đây là bắt buộc:

### 5.1. Khử Lệch Bằng Log-Transformation (`log1p`)
Do phân bố lỗi bị lệch phải cực đoan, ta áp dụng hàm Log-transform cho nhãn mục tiêu:
$$y_{log} = \ln(y_{original} + 1)$$
Hàm `log1p` giúp kéo các giá trị cực trị về gần trung tâm hơn, biến đổi phân bố lệch trở nên đối xứng hơn. Việc cộng thêm 1 nhằm đảm bảo lớp không lỗi ($y=0$) vẫn giữ nguyên giá trị 0 sau khi log: $\ln(0+1) = 0$. 

*Khi đánh giá dự báo, ta khôi phục về đơn vị lỗi gốc bằng hàm ngược:*
$$y_{original\_pred} = e^{y_{log\_pred}} - 1$$

### 5.2. Chuẩn Hóa Đặc Trưng (Z-score Standardization)
Các metric phần mềm có thang đo chênh lệch lớn (ví dụ: `dit` chạy từ 1-5, còn `loc` lên đến hàng nghìn). Ta áp dụng chuẩn hóa Standardizer:
$$X_{standardized} = \frac{X - \mu}{\sigma}$$
Đưa mọi đặc trưng về phân bố có trung bình bằng 0 và độ lệch chuẩn bằng 1, giúp thuật toán tối ưu hóa Gradient Descent hội tụ nhanh và ổn định hơn rất nhiều.

### 5.3. Kỹ Thuật SMOTEND Cho Hồi Quy
Dữ liệu PROMISE có sự mất cân bằng nghiêm trọng giữa các mẫu ít lỗi và nhiều lỗi. Kỹ thuật **SMOTEND (Synthetic Minority Over-sampling Technique for Regression with Continuous Target)** tự định nghĩa hoạt động như sau:
1.  Xác định tập các lớp thiểu số có nguy cơ lỗi cao ($y > 0$).
2.  Với mỗi điểm dữ liệu có lỗi, tính toán $k$ láng giềng gần nhất của nó trong không gian đặc trưng.
3.  Chọn ngẫu nhiên một láng giềng và sinh mẫu mới bằng cách nội suy tuyến tính:
    $$X_{new} = X_{current} + \lambda \times (X_{neighbor} - X_{current})$$
    với $\lambda \in [0, 1]$ là số ngẫu nhiên.
4.  Nội suy nhãn mục tiêu lỗi tương ứng giúp tạo ra nhãn lỗi tổng hợp mượt mà và thực tế hơn:
    $$y_{new} = y_{current} + \lambda \times (y_{neighbor} - y_{current})$$

Nhờ SMOTEND, mô hình học máy học được phân bố của các lớp bị lỗi nhiều tốt hơn, cải thiện đáng kể hệ số tương quan thứ hạng Kendall trên tập kiểm thử độc lập.
