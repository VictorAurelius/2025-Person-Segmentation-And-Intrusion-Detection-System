PHẦN MỞ ĐẦU


================================================================================


                           ĐẠI HỌC [TÊN TRƯỜNG]
                            KHOA [TÊN KHOA]


                                  ---


                        BÁO CÁO ĐỒ ÁN MÔN HỌC

                              XỬ LÝ ẢNH


                                  ---


                                ĐỀ TÀI:
          PHÂN VÙNG NGƯỜI VÀ PHÁT HIỆN XÂM NHẬP KHU VỰC CẤM


                                  ---


                    SINH VIÊN THỰC HIỆN: [Họ và tên]
                          MSSV: [Mã số sinh viên]
                             LỚP: [Tên lớp]

               GIẢNG VIÊN HƯỚNG DẪN: [Họ và tên giảng viên]


                                  ---


                  Thành phố Hồ Chí Minh, tháng 11 năm 2025


================================================================================




LỜI CẢM ƠN


     Em xin chân thành cảm ơn [Tên giảng viên hướng dẫn], người đã tận tình hướng dẫn, giúp đỡ em trong suốt quá trình thực hiện đồ án này. Những góp ý chuyên môn sâu sắc, những lời khuyên định hướng của thầy cô đã giúp em vượt qua nhiều khó khăn kỹ thuật và hiểu rõ hơn về các kỹ thuật xử lý ảnh trong thực tế. Sự kiên nhẫn và tận tâm của thầy cô trong việc giải đáp thắc mắc, review code, và góp ý cải tiến là nguồn động lực lớn giúp em hoàn thiện đồ án.

     Em cũng xin gửi lời cảm ơn đến gia đình và bạn bè đã luôn động viên, khuyến khích em trong suốt quá trình học tập và nghiên cứu. Sự ủng hộ tinh thần từ mọi người đã giúp em vượt qua những lúc khó khăn, đặc biệt trong giai đoạn debug và tối ưu hóa hệ thống.

     Em xin cảm ơn cộng đồng OpenCV và các nhà nghiên cứu trên thế giới đã công bố các papers, tutorials, và documentation chất lượng cao. Những tài liệu này đã là nền tảng quan trọng giúp em tiếp cận với state-of-the-art techniques trong xử lý ảnh và computer vision. Đặc biệt, em biết ơn các tác giả đã chia sẻ implementation code và best practices một cách miễn phí.

     Cuối cùng, em xin gửi lời cảm ơn đến các nhà phát triển dataset công khai như VIRAT, CAVIAR, và ChokePoint đã tạo điều kiện cho sinh viên và nhà nghiên cứu có thể truy cập dữ liệu chất lượng để test và đánh giá hệ thống. Sự đóng góp của họ cho cộng đồng nghiên cứu là vô cùng quý báu.

     Mặc dù đã nỗ lực hết mình, đồ án này không thể tránh khỏi những thiếu sót. Em rất mong nhận được sự góp ý, chỉ bảo từ thầy cô và các bạn để có thể tiếp tục hoàn thiện và phát triển hệ thống trong tương lai.


                                        Thành phố Hồ Chí Minh, tháng 11 năm 2025
                                                      Sinh viên thực hiện
                                                       [Họ và tên]


================================================================================




TÓM TẮT


     Phát hiện người xâm nhập vùng cấm là một bài toán quan trọng trong lĩnh vực giám sát an ninh tự động, được ứng dụng rộng rãi tại các sân bay, nhà máy, khu công nghiệp, và không gian công cộng. Việc giám sát bằng con người liên tục 24/7 không chỉ tốn kém chi phí nhân lực mà còn dễ dẫn đến sai sót do mệt mỏi và thiếu tập trung. Đồ án này nhằm xây dựng một hệ thống tự động phát hiện và cảnh báo khi có người xâm nhập vào khu vực bị hạn chế, sử dụng các kỹ thuật xử lý ảnh số truyền thống kết hợp với computer vision.

     Hệ thống được phát triển dựa trên pipeline xử lý ảnh modular kết hợp nhiều kỹ thuật bổ trợ lẫn nhau. Đầu tiên, Motion Detection được thực hiện bằng Background Subtraction với ba phương pháp: MOG2 (Mixture of Gaussians), KNN (K-Nearest Neighbors), và Frame Differencing, trong đó MOG2 được chọn làm phương pháp chính nhờ khả năng thích ứng tốt với thay đổi ánh sáng và phát hiện bóng đổ. Thứ hai, Adaptive Thresholding và CLAHE (Contrast Limited Adaptive Histogram Equalization) được áp dụng để xử lý điều kiện ánh sáng không đồng đều và cải thiện độ tương phản trong môi trường thiếu sáng. Thứ ba, Edge Detection bằng thuật toán Canny giúp phát hiện đường viền chính xác của đối tượng chuyển động. Thứ tư, Region Growing và Contour Analysis được sử dụng để phân vùng và tìm bounding box của từng đối tượng. Cuối cùng, Intrusion Detection dựa trên tính toán IoU (Intersection over Union) giữa bounding box và ROI (Region of Interest) polygon, kết hợp với time-based validation để giảm false positives từ chuyển động tạm thời.

     Vùng cấm được định nghĩa linh hoạt bằng polygon tùy chỉnh thông qua công cụ ROI Selector tương tác, cho phép người dùng vẽ trực tiếp trên video mà không cần kiến thức lập trình. Hệ thống hỗ trợ nhiều ROI đồng thời và lưu trữ dưới dạng JSON dễ đọc, dễ chỉnh sửa. Alert System được thiết kế đa dạng với visual alert (banner cảnh báo màu đỏ trên video), audio alert (âm thanh cảnh báo), logging chi tiết vào file (timestamp, ROI name, duration, location, screenshot path), và tự động lưu screenshots cho mỗi event để phục vụ audit. Cooldown mechanism ngăn spam alerts khi một đối tượng ở lại trong ROI lâu.

     Kết quả thực nghiệm trên hai video test đại diện cho hai điều kiện môi trường khác nhau cho thấy hệ thống đạt hiệu năng khả quan. Detection rate đạt 81.7 phần trăm trung bình, gần sát mục tiêu 85 phần trăm, với input-01 (cảnh ban ngày ngoài trời) đạt 80 phần trăm và input-02 (cảnh trong nhà) đạt 83.3 phần trăm. False positive rate chỉ 8.3 phần trăm, nằm trong ngưỡng chấp nhận được dưới 10 phần trăm. Tốc độ xử lý đạt 26.6 FPS trung bình, vượt mục tiêu 20 FPS, cho phép xử lý video gần real-time với tốc độ chỉ chậm hơn video gốc 1.03 lần. Precision đạt 90 phần trăm, Recall đạt 81.8 phần trăm, và F1-Score đạt 85.7 phần trăm, phản ánh sự cân bằng tốt giữa độ chính xác và độ phủ. Hệ thống chạy ổn định trên hardware phổ thông (CPU Intel Core i5 hoặc tương đương) mà không cần GPU chuyên dụng.

     Code được tổ chức theo kiến trúc modular với các module độc lập (MotionDetector, AdaptiveThreshold, EdgeDetector, IntrusionDetector, AlertSystem), giúp dễ đọc, dễ test, dễ maintain, và dễ mở rộng. File cấu hình YAML tập trung cho phép tuning tham số mà không cần sửa code. Hệ thống tuân theo software engineering best practices, tạo nền tảng tốt cho phát triển lâu dài.

     Hệ thống có tiềm năng ứng dụng cao trong thực tế với chi phí triển khai hợp lý (1000-2500 USD mỗi camera setup) và ROI (Return on Investment) nhanh (3-6 tháng) nhờ tiết kiệm chi phí nhân lực giám sát 24/7. Các lĩnh vực ứng dụng bao gồm an ninh công cộng (sân bay, nhà ga, bến xe), an ninh doanh nghiệp (nhà máy, kho hàng, server room), giám sát giao thông (đường ray, làn ô tô), và smart home security. Hệ thống có khả năng mở rộng dễ dàng bằng cách thêm cameras và instances mà không cần thay đổi infrastructure lớn.


================================================================================




MỤC LỤC


PHẦN MỞ ĐẦU
   Lời cảm ơn ................................................... [trang]
   Tóm tắt ...................................................... [trang]
   Mục lục ...................................................... [trang]
   Danh sách hình ............................................... [trang]
   Danh sách bảng ............................................... [trang]


CHƯƠNG 1: CƠ SỞ LÝ THUYẾT ....................................... [trang]
   1.1. Tổng Quan Về Xử Lý Ảnh Số .............................. [trang]
        1.1.1. Định Nghĩa Ảnh Số ................................ [trang]
        1.1.2. Biểu Diễn Ảnh Số ................................. [trang]
        1.1.3. Color Spaces ...................................... [trang]
        1.1.4. Vai Trò Trong Giám Sát An Ninh ................... [trang]

   1.2. Motion Detection - Phát Hiện Chuyển Động ............... [trang]
        1.2.1. Frame Differencing ................................ [trang]
        1.2.2. Background Subtraction ............................ [trang]
        1.2.3. MOG2 (Mixture of Gaussians) ....................... [trang]
        1.2.4. KNN Background Subtractor ......................... [trang]
        1.2.5. So Sánh Các Phương Pháp ........................... [trang]

   1.3. Adaptive Thresholding - Ngưỡng Hóa Thích Ứng ........... [trang]
        1.3.1. Global vs Adaptive Thresholding ................... [trang]
        1.3.2. Gaussian Adaptive Threshold ....................... [trang]
        1.3.3. Mean Adaptive Threshold ........................... [trang]
        1.3.4. CLAHE - Contrast Limited AHE ...................... [trang]

   1.4. Edge Detection - Phát Hiện Biên ......................... [trang]
        1.4.1. Khái Niệm Gradient và Edge ....................... [trang]
        1.4.2. Canny Edge Detection .............................. [trang]
        1.4.3. Sobel Operator .................................... [trang]
        1.4.4. So Sánh Canny vs Sobel ............................ [trang]

   1.5. Region Growing - Mở Rộng Vùng ........................... [trang]
        1.5.1. Thuật Toán Region Growing ......................... [trang]
        1.5.2. Tiêu Chí Tương Đồng ............................... [trang]
        1.5.3. Ứng Dụng Trong Segmentation ....................... [trang]

   1.6. Intrusion Detection - Phát Hiện Xâm Nhập ............... [trang]
        1.6.1. ROI - Region of Interest .......................... [trang]
        1.6.2. IoU - Intersection over Union ..................... [trang]
        1.6.3. Time-based Validation ............................. [trang]
        1.6.4. Tracking Mechanism ................................ [trang]

   1.7. Các Yếu Tố Ảnh Hưởng Đến Chất Lượng .................... [trang]
        1.7.1. Độ Phân Giải ...................................... [trang]
        1.7.2. Điều Kiện Ánh Sáng ................................ [trang]
        1.7.3. Nhiễu và Chất Lượng Ảnh ........................... [trang]
        1.7.4. Chuyển Động Camera ................................ [trang]


CHƯƠNG 2: CƠ SỞ THỰC HÀNH ....................................... [trang]
   2.1. Quy Trình Thu Thập và Chuẩn Bị Dữ Liệu ................. [trang]
        2.1.1. Tiêu Chí Lựa Chọn Dữ Liệu ........................ [trang]
        2.1.2. Nguồn Dữ Liệu .................................... [trang]
        2.1.3. Tiền Xử Lý Dữ Liệu ............................... [trang]
        2.1.4. Định Nghĩa ROI .................................... [trang]

   2.2. Kiến Trúc Hệ Thống ...................................... [trang]
        2.2.1. Tổng Quan Kiến Trúc ............................... [trang]
        2.2.2. Các Module Chính .................................. [trang]
        2.2.3. Luồng Xử Lý Từng Frame ............................ [trang]

   2.3. Phân Tích Chi Tiết Các Kỹ Thuật Áp Dụng ................ [trang]
        2.3.1. Motion Detection Module ........................... [trang]
        2.3.2. Adaptive Thresholding Module ...................... [trang]
        2.3.3. Edge Detection Module ............................. [trang]
        2.3.4. Intrusion Detection Module ........................ [trang]
        2.3.5. Alert System Module ............................... [trang]

   2.4. Cấu Hình và Tối Ưu Tham Số ............................. [trang]
        2.4.1. Cấu Trúc File Config .............................. [trang]
        2.4.2. Tuning Theo Điều Kiện Ánh Sáng .................... [trang]
        2.4.3. Thực Nghiệm Tuning ................................ [trang]

   2.5. Quy Trình Thực Thi Hệ Thống ............................ [trang]
        2.5.1. Setup Môi Trường .................................. [trang]
        2.5.2. Chạy Hệ Thống ..................................... [trang]
        2.5.3. Output của Hệ Thống ............................... [trang]

   2.6. Đánh Giá Kết Quả Thực Nghiệm ............................ [trang]
        2.6.1. Test Scenarios .................................... [trang]
        2.6.2. Performance Metrics ............................... [trang]
        2.6.3. Kết Quả Thực Tế ................................... [trang]
        2.6.4. Phân Tích Alerts .................................. [trang]

   2.7. So Sánh Với Các Phương Pháp Khác ........................ [trang]
        2.7.1. So Sánh Với Baseline Methods ...................... [trang]
        2.7.2. Ưu Điểm của Hệ Thống Hiện Tại .................... [trang]
        2.7.3. Nhược Điểm và Hạn Chế ............................. [trang]


CHƯƠNG 3: KẾT LUẬN VÀ ĐÁNH GIÁ ................................. [trang]
   3.1. Tóm Tắt Kết Quả Đạt Được ................................ [trang]
        3.1.1. Mục Tiêu Đã Hoàn Thành ............................ [trang]
        3.1.2. Kết Quả Định Lượng ................................ [trang]
        3.1.3. Đầu Ra của Hệ Thống ............................... [trang]

   3.2. Đánh Giá Hiệu Quả và Độ Chính Xác ....................... [trang]
        3.2.1. Điểm Mạnh ......................................... [trang]
        3.2.2. Điểm Yếu .......................................... [trang]
        3.2.3. Độ Tin Cậy ........................................ [trang]
        3.2.4. So Sánh Mục Tiêu Với Kết Quả Thực Tế .............. [trang]

   3.3. Đề Xuất Cải Tiến ........................................ [trang]
        3.3.1. Cải Tiến Về Tốc Độ Xử Lý ......................... [trang]
        3.3.2. Cải Tiến Về Độ Nhạy ............................... [trang]
        3.3.3. Xử Lý Các Tình Huống Đặc Biệt .................... [trang]
        3.3.4. Mở Rộng Chức Năng ................................. [trang]

   3.4. Ứng Dụng Thực Tế ........................................ [trang]
        3.4.1. Các Lĩnh Vực Ứng Dụng ............................. [trang]
        3.4.2. Triển Khai Thực Tế ................................ [trang]
        3.4.3. Chi Phí và Lợi Ích ................................ [trang]

   3.5. Kết Luận Chung .......................................... [trang]
        3.5.1. Tổng Kết .......................................... [trang]
        3.5.2. Ý Nghĩa ........................................... [trang]
        3.5.3. Hướng Phát Triển Tương Lai ........................ [trang]
        3.5.4. Lời Cảm Ơn ........................................ [trang]


TÀI LIỆU THAM KHẢO .............................................. [trang]


PHỤ LỤC ......................................................... [trang]
   A. Source Code Chính ......................................... [trang]
   B. File Cấu Hình ............................................. [trang]
   C. ROI Definitions ........................................... [trang]
   D. Sample Alerts Log ......................................... [trang]


================================================================================




DANH SÁCH HÌNH


CHƯƠNG 1: CƠ SỞ LÝ THUYẾT

Hình 1.1:  Biểu diễn ảnh số dưới dạng ma trận pixel ................ [trang]
Hình 1.2:  Các color space phổ biến (RGB, Grayscale, HSV) ......... [trang]
Hình 1.3:  Minh họa Frame Differencing trên 3 frames liên tiếp .... [trang]
Hình 1.4:  Quá trình Background Subtraction với MOG2 .............. [trang]
Hình 1.5:  Ví dụ foreground mask từ MOG2 trên video surveillance .. [trang]
Hình 1.6:  So sánh Global Thresholding vs Adaptive Thresholding ... [trang]
Hình 1.7:  Kết quả CLAHE trên ảnh thiếu sáng ....................... [trang]
Hình 1.8:  5 bước của thuật toán Canny Edge Detection .............. [trang]
Hình 1.9:  So sánh kết quả Canny vs Sobel edge detection .......... [trang]
Hình 1.10: Quá trình Region Growing từ seed points ................. [trang]
Hình 1.11: Ví dụ ROI polygon trên video giám sát ................... [trang]
Hình 1.12: Minh họa tính toán IoU giữa bbox và ROI ................. [trang]
Hình 1.13: Quy trình phát hiện xâm nhập hoàn chỉnh ................. [trang]


CHƯƠNG 2: CƠ SỞ THỰC HÀNH

Hình 2.1:  Giao diện ROI Selector Tool với video input-01 ......... [trang]
Hình 2.2:  Ví dụ ROI polygon hoàn chỉnh trên video test ............ [trang]
Hình 2.3:  Sơ đồ kiến trúc hệ thống tổng thể ....................... [trang]
Hình 2.4:  Flowchart xử lý một frame ................................ [trang]
Hình 2.5:  So sánh foreground mask từ 3 phương pháp ................ [trang]
Hình 2.6:  Kết quả adaptive thresholding trên frame thiếu sáng .... [trang]
Hình 2.7:  Edge detection kết hợp với motion mask .................. [trang]
Hình 2.8:  Ví dụ tính toán IoU giữa bounding box và ROI ............ [trang]
Hình 2.9:  Screenshot terminal khi chạy hệ thống ................... [trang]
Hình 2.10: Cấu trúc thư mục output .................................. [trang]
Hình 2.11: Frame có alert từ video input-01 (alert_0001.jpg) ...... [trang]
Hình 2.12: Frame có alert từ video input-02 (alert_0003.jpg) ...... [trang]
Hình 2.13: Biểu đồ phân bố alerts theo thời gian ................... [trang]


CHƯƠNG 3: KẾT LUẬN VÀ ĐÁNH GIÁ

Hình 3.1:  Frame output hoàn chỉnh với tất cả annotations .......... [trang]
Hình 3.2:  Ví dụ screenshot alert có info overlay .................. [trang]
Hình 3.3:  Sơ đồ kiến trúc mở rộng với GPU và multi-threading ..... [trang]
Hình 3.4:  Flowchart xử lý occlusion với Kalman filter ............. [trang]


Tổng số: 30 hình


Lưu ý: Số trang sẽ được cập nhật tự động khi tạo danh sách hình trong Microsoft Word sử dụng tính năng Table of Figures.


================================================================================




DANH SÁCH BẢNG


CHƯƠNG 1: CƠ SỞ LÝ THUYẾT

Bảng 1.1: So sánh Frame Differencing vs Background Subtraction ..... [trang]
Bảng 1.2: Ảnh hưởng của các yếu tố môi trường đến hiệu năng ....... [trang]


CHƯƠNG 2: CƠ SỞ THỰC HÀNH

Bảng 2.1: Thông số kỹ thuật các video test ......................... [trang]
Bảng 2.2: So sánh hiệu năng các phương pháp motion detection ...... [trang]
Bảng 2.3: Tham số tối ưu cho các điều kiện ánh sáng ................ [trang]
Bảng 2.4: Kết quả thực nghiệm tuning tham số motion.threshold ..... [trang]
Bảng 2.5: Tổng hợp kết quả trên 2 video test ....................... [trang]
Bảng 2.6: Confusion Matrix tổng hợp cho cả 2 video ................. [trang]
Bảng 2.7: So sánh metrics với các phương pháp khác ................. [trang]


CHƯƠNG 3: KẾT LUẬN VÀ ĐÁNH GIÁ

Bảng 3.1: Tổng hợp kết quả định lượng từ thực nghiệm ............... [trang]
Bảng 3.2: So sánh mục tiêu với kết quả thực tế ..................... [trang]


Tổng số: 11 bảng


Lưu ý: Số trang sẽ được cập nhật tự động khi tạo danh sách bảng trong Microsoft Word sử dụng tính năng Table of Tables.


================================================================================
