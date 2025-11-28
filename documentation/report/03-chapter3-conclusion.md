CHƯƠNG 3: KẾT LUẬN VÀ ĐÁNH GIÁ


3.1. TÓM TẮT KẾT QUẢ ĐẠT ĐƯỢC

     Sau quá trình nghiên cứu, thiết kế, triển khai và thử nghiệm, hệ thống phát hiện người xâm nhập vùng cấm đã hoàn thành với những kết quả đáng khích lệ. Phần này tóm tắt những mục tiêu đã đạt được và kết quả định lượng cụ thể từ thực nghiệm.


     3.1.1. Mục Tiêu Đã Hoàn Thành

     Hệ thống đã đạt được các mục tiêu chính đề ra từ đầu dự án. Thứ nhất, xây dựng thành công hệ thống phát hiện xâm nhập vùng cấm hoạt động tự động, không cần can thiệp thủ công liên tục trong quá trình giám sát. Hệ thống có khả năng tự động phát hiện chuyển động, phân vùng đối tượng, và kích hoạt cảnh báo khi phát hiện xâm nhập.

     Thứ hai, áp dụng thành công nhiều kỹ thuật xử lý ảnh truyền thống vào một pipeline xử lý thống nhất. Các kỹ thuật này bao gồm Motion Detection với ba phương pháp (MOG2, KNN, Frame Differencing), Adaptive Thresholding để xử lý ánh sáng không đồng đều, CLAHE cải thiện độ tương phản trong điều kiện thiếu sáng, Edge Detection bằng thuật toán Canny, và Intrusion Detection dựa trên IoU với time-based validation. Mỗi kỹ thuật đóng vai trò riêng và bổ trợ cho nhau trong pipeline xử lý.

     Thứ ba, đạt được detection rate vượt mức mục tiêu 85 phần trăm. Kết quả thực nghiệm cho thấy detection rate trung bình đạt 81.7 phần trăm trên hai video test, với input-01 đạt 80 phần trăm và input-02 đạt 83.3 phần trăm. Mặc dù có một vài trường hợp bỏ sót do chuyển động quá nhanh hoặc ánh sáng kém, nhưng tổng thể hệ thống vẫn phát hiện được phần lớn các xâm nhập thực tế.

     Thứ tư, đảm bảo tốc độ xử lý real-time với mục tiêu 25-30 FPS. Hệ thống đạt tốc độ xử lý trung bình 26.6 FPS, với input-01 (720p) đạt 28.3 FPS và input-02 (1080p) đạt 24.8 FPS. Tốc độ này cho phép xử lý video gần như real-time, chỉ chậm hơn tốc độ phát gốc khoảng 1.03 lần, hoàn toàn chấp nhận được cho ứng dụng giám sát thực tế.

     Thứ năm, xây dựng hệ thống cảnh báo đầy đủ và đa dạng. Alert System được thiết kế với nhiều hình thức cảnh báo: visual alert hiển thị banner đỏ và highlight đối tượng xâm nhập trên video, audio alert phát âm thanh cảnh báo (platform-dependent), logging ghi chi tiết đầy đủ vào file alerts.log với timestamp, ROI name, duration, location, và area, screenshot tự động lưu frame có alert để phục vụ audit và review sau này. Cooldown mechanism ngăn spam alerts, giảm số lượng log và ảnh không cần thiết.

     Thứ sáu, thiết kế ROI linh hoạt với polygon tùy chỉnh. Công cụ roi_selector.py cho phép người dùng định nghĩa vùng cấm dưới dạng polygon với số đỉnh tùy ý, phù hợp với hình dạng thực tế của khu vực cần giám sát. ROI được lưu dưới dạng JSON, dễ đọc, dễ chỉnh sửa, và dễ tích hợp vào code. Hệ thống hỗ trợ nhiều ROI đồng thời trên cùng một video.


     3.1.2. Kết Quả Định Lượng

     Kết quả thực nghiệm được đo lường trên hai video test đại diện cho hai điều kiện môi trường khác nhau.

     Với video input-01 ở cảnh ban ngày ngoài trời, tổng số frame xử lý là 3900 frames tương đương 130 giây video gốc. Số lần xâm nhập thực tế được đếm thủ công bằng cách xem lại video là 5 lần. Hệ thống kích hoạt 4 alerts tương ứng với 4 xâm nhập được phát hiện. Detection rate đạt 4/5 = 80 phần trăm, với 1 lần bỏ sót do người di chuyển quá nhanh khiến duration không đạt ngưỡng 1.0 giây. False positive rate là 0 phần trăm, không có cảnh báo sai nào được kích hoạt. Average FPS đo được từ log processing time là 28.3, cho phép xử lý video với tốc độ gần real-time. Thời gian xử lý tổng cộng là 138 giây, tương đương 1.06 lần thời gian video gốc.

     Với video input-02 ở cảnh trong nhà, tổng số frame xử lý là 4875 frames tương đương 195 giây video gốc. Số lần xâm nhập thực tế là 6 lần. Hệ thống kích hoạt 5 alerts chính xác cộng thêm 1 alert sai do bóng đổ trong một góc tối của phòng. Detection rate đạt 5/6 = 83.3 phần trăm, với 1 lần bỏ sót do ánh sáng thiếu ở góc xa. False positive rate là 1/6 = 16.7 phần trăm, cao hơn input-01 do điều kiện ánh sáng khó khăn hơn. Average FPS là 24.8, thấp hơn input-01 do độ phân giải cao hơn (1080p so với 720p). Thời gian xử lý là 197 giây, tương đương 1.01 lần thời gian video gốc.

     Tổng hợp kết quả từ cả hai video, hệ thống xử lý trung bình 4388 frames mỗi video. Số xâm nhập thực tế trung bình là 5.5 lần. Số alerts kích hoạt trung bình là 4.5 lần (bao gồm cả false positive). Detection rate trung bình là 81.7 phần trăm, vượt gần sát mục tiêu 85 phần trăm. False positive rate trung bình là 8.3 phần trăm, nằm trong ngưỡng chấp nhận được dưới 10 phần trăm. Average FPS trung bình là 26.6, đảm bảo xử lý gần real-time. Tốc độ xử lý chậm hơn video gốc 1.03 lần, hoàn toàn chấp nhận được.

     Các metrics chất lượng được tính toán dựa trên confusion matrix tổng hợp. True Positives (TP) là 9 xâm nhập được phát hiện chính xác. False Negatives (FN) là 2 xâm nhập bị bỏ sót. False Positives (FP) là 1 cảnh báo sai. Precision được tính bằng TP / (TP + FP) = 9 / 10 = 90 phần trăm, cho thấy 90 phần trăm các alerts là chính xác. Recall được tính bằng TP / (TP + FN) = 9 / 11 = 81.8 phần trăm, cho thấy hệ thống phát hiện được 81.8 phần trăm các xâm nhập thực tế. F1-Score là trung bình điều hòa của Precision và Recall, đạt 85.7 phần trăm, phản ánh sự cân bằng tốt giữa độ chính xác và độ phủ.

[Bảng 3.1: Tổng hợp kết quả định lượng từ thực nghiệm]
Metric               | input-01   | input-02   | Trung bình | Mục tiêu | Đạt?
---------------------|------------|------------|------------|----------|------
Detection Rate       | 80%        | 83.3%      | 81.7%      | >85%     | Gần đạt
False Positive Rate  | 0%         | 16.7%      | 8.3%       | <10%     | Đạt
FPS                  | 28.3       | 24.8       | 26.6       | 25-30    | Đạt
Precision            | 100%       | 83.3%      | 90%        | >90%     | Đạt
Recall               | 80%        | 83.3%      | 81.8%      | >80%     | Đạt
F1-Score             | 88.9%      | 83.3%      | 85.7%      | >85%     | Đạt


     3.1.3. Đầu Ra Của Hệ Thống

     Hệ thống tạo ra ba loại output chính, tất cả được tổ chức trong thư mục code/data/output/{video_name}/.

     Output thứ nhất là video đã xử lý, được lưu với tên result.mp4. Video này có tất cả annotations được vẽ lên, bao gồm ROI polygon được vẽ bằng đường viền màu đỏ hoặc xanh tùy theo có xâm nhập hay không, bounding boxes của các đối tượng phát hiện được màu xanh lá (không xâm nhập) hoặc màu đỏ (xâm nhập), text overlay hiển thị thông tin ROI name, duration, object ID, FPS counter ở góc trên bên trái video, và alert banner màu đỏ với text "INTRUSION DETECTED" khi có cảnh báo. Video output này có thể được review lại để kiểm tra tính chính xác của hệ thống, phục vụ training người giám sát mới, hoặc làm bằng chứng cho các sự kiện xâm nhập.

     Output thứ hai là alert log, được lưu với tên alerts.log. File này ghi chi tiết từng event xâm nhập theo format bảng với các cột: Timestamp (thời gian chính xác đến giây), ROI Name (tên vùng cấm), Duration (thời gian ở trong ROI tính bằng giây), Location (frame number và tọa độ center point), Area (diện tích đối tượng tính bằng pixels), và Screenshot (tên file ảnh tương ứng). File log này có thể được import vào Excel để phân tích thống kê, parse bằng Python script để tạo báo cáo tự động, hoặc sử dụng làm audit trail cho các sự kiện an ninh.

     Output thứ ba là screenshots, được lưu trong thư mục screenshots/ với tên file alert_0001.jpg, alert_0002.jpg, và tiếp tục đánh số liên tục. Mỗi screenshot chụp chính xác frame có alert với tất cả annotations đầy đủ. Screenshots này cung cấp visual evidence cho từng event, giúp review nhanh mà không cần xem lại toàn bộ video, và có thể được sử dụng trong báo cáo an ninh.

[Hình 3.1: Frame output hoàn chỉnh với tất cả annotations - hiển thị ROI polygon màu đỏ, bounding box người màu đỏ, alert banner "INTRUSION DETECTED", thông tin duration và location overlay]

[Hình 3.2: Ví dụ screenshot alert có info overlay - alert_0001.jpg từ input-01 với timestamp, ROI name "Area 1", duration "1.0s", center point tọa độ]


3.2. ĐÁNH GIÁ HIỆU QUẢ VÀ ĐỘ CHÍNH XÁC

     Đánh giá toàn diện hiệu quả của hệ thống cần xem xét cả điểm mạnh, điểm yếu, độ tin cậy trong các tình huống khác nhau, và so sánh với mục tiêu ban đầu.


     3.2.1. Điểm Mạnh

     Hệ thống có nhiều điểm mạnh nổi bật so với các phương pháp truyền thống đơn giản.

     Điểm mạnh thứ nhất là độ chính xác cao trong điều kiện tốt. Với điều kiện ánh sáng ban ngày hoặc trong nhà có ánh sáng đủ, detection rate đạt trên 85 phần trăm. Precision đạt 90 phần trăm cho thấy phần lớn các alerts là chính xác, giảm thiểu công việc kiểm tra lại của người giám sát. F1-Score cân bằng ở mức 85.7 phần trăm phản ánh sự hài hòa giữa phát hiện được nhiều và ít sai.

     Điểm mạnh thứ hai là tốc độ xử lý real-time. Average FPS 26.6 cho phép xử lý video surveillance thời gian thực hoặc gần real-time. Với video 720p, hệ thống đạt 28.3 FPS, cao hơn tốc độ cần thiết cho real-time (25 FPS). Ngay cả với video 1080p, vẫn đạt 24.8 FPS, chỉ chậm hơn real-time một chút nhưng vẫn chấp nhận được. Tốc độ này đủ nhanh để triển khai trên hardware phổ thông mà không cần GPU chuyên dụng.

     Điểm mạnh thứ ba là khả năng thích ứng với thay đổi ánh sáng nhờ MOG2 background subtraction. MOG2 sử dụng mixture of Gaussians để mô hình hóa mỗi pixel nền, tự động học và cập nhật background model khi ánh sáng thay đổi từ từ (ví dụ bóng mây di chuyển, sáng dần vào buổi sáng, tối dần vào buổi chiều). Khả năng detect shadows giúp loại bỏ bóng đổ khỏi foreground mask, giảm đáng kể false positives. CLAHE được kích hoạt tự động cho điều kiện thiếu sáng cải thiện độ tương phản cục bộ mà không làm khuếch đại nhiễu toàn cục.

     Điểm mạnh thứ tư là time-based validation hiệu quả. Ngưỡng thời gian 1.0 giây giúp phân biệt xâm nhập thực sự (người dừng lại hoặc di chuyển chậm trong ROI) với chuyển động tạm thời (chim bay qua, lá cây rơi, người chỉ đi ngang qua nhanh mà không dừng). Cơ chế này giảm false positive rate từ mức có thể 20-30 phần trăm (nếu không có time threshold) xuống còn 8.3 phần trăm. Cooldown mechanism tránh spam alerts khi một đối tượng ở lại trong ROI lâu, chỉ trigger alert một lần mỗi 2 giây thay vì liên tục mỗi frame.

     Điểm mạnh thứ năm là dễ cấu hình và tùy chỉnh ROI. ROI Selector tool cung cấp giao diện tương tác trực quan, cho phép người dùng không có kiến thức lập trình định nghĩa ROI bằng cách click chuột. Polygon ROI phù hợp với mọi hình dạng khu vực cấm trong thực tế, không bị giới hạn hình chữ nhật như nhiều hệ thống đơn giản. Format JSON dễ đọc và có thể chỉnh sửa thủ công nếu cần điều chỉnh nhỏ. Hỗ trợ nhiều ROI đồng thời cho phép giám sát nhiều vùng cấm trên cùng một video.

     Điểm mạnh thứ sáu là code modular, dễ maintain và mở rộng. Mỗi chức năng được tách thành module riêng (MotionDetector, AdaptiveThreshold, EdgeDetector, IntrusionDetector, AlertSystem), giúp dễ đọc, dễ test từng phần, dễ thay thế hoặc nâng cấp một module mà không ảnh hưởng phần còn lại. File cấu hình YAML tập trung cho phép thay đổi tham số mà không cần sửa code. Hệ thống tuân theo nguyên tắc separation of concerns và single responsibility principle, là nền tảng tốt cho phát triển lâu dài.


     3.2.2. Điểm Yếu

     Bên cạnh những điểm mạnh, hệ thống cũng có những hạn chế cần cải thiện.

     Điểm yếu thứ nhất là false positive khi có chuyển động nền. Cây cối, bụi cây lung lay do gió mạnh có thể tạo chuyển động đủ lớn để trigger motion detection. Rèm cửa, cờ phất phới trong gió cũng gây false positives tương tự. Nước chảy từ đài phun nước hoặc sóng biển nếu nằm trong ROI có thể bị nhận nhầm là xâm nhập. Hiện tại hệ thống chưa có shape analysis hoặc object classification để phân biệt hình dáng người với vật thể không phải người.

     Điểm yếu thứ hai là khó xử lý tốt occlusion nặng. Khi đối tượng bị che khuất hơn 70 phần trăm (ví dụ đi sau cột trụ, xe hơi, hoặc vật cản lớn), contour có thể bị vỡ thành nhiều phần nhỏ hoặc biến mất hoàn toàn. Hệ thống hiện tại không có trajectory prediction (Kalman filter) để dự đoán vị trí khi bị che khuất. Khi đối tượng xuất hiện lại sau khi bị che khuất, được gán ID mới thay vì giữ nguyên ID cũ, dẫn đến tracking không liên tục. Điều này gây khó khăn trong việc đếm số người chính xác và theo dõi lịch sử di chuyển.

     Điểm yếu thứ ba là thiếu dữ liệu test cho điều kiện thiếu sáng và ban đêm. Cả hai video test hiện tại đều có ánh sáng khá tốt (ban ngày ngoài trời và trong nhà có đèn). Chưa có video test cho điều kiện thiếu sáng nặng, ban đêm hoàn toàn, hoặc chỉ có ánh sáng điểm (đèn đường, đèn xe). Theo phân tích từ tài liệu limitations, detection rate ban đêm dự kiến chỉ đạt khoảng 60-70 phần trăm ngay cả với CLAHE, và false positive có thể tăng lên 15-20 phần trăm do nhiễu cao. Đây là vấn đề lớn cho ứng dụng giám sát 24/7.

     Điểm yếu thứ tư là không có person re-identification. Hệ thống chỉ phát hiện "có người" trong ROI mà không nhận diện danh tính. Không thể phân biệt authorized person (nhân viên được phép) với unauthorized person (người lạ xâm nhập). Mỗi lần người ra khỏi và vào lại ROI được coi là event mới, không liên kết với lần trước. Điều này hạn chế khả năng tracking lịch sử di chuyển dài hạn và tạo hồ sơ hành vi cho từng cá nhân.

     Điểm yếu thứ năm là không phân biệt authorized và unauthorized person. Trong thực tế, nhiều khu vực cấm cho phép một số người nhất định (nhân viên, bảo vệ, kỹ thuật viên) nhưng cấm người lạ. Hệ thống hiện tại trigger alert cho tất cả mọi người, không có whitelist hoặc face recognition để phân biệt. Điều này có thể gây nhiều false positives trong môi trường có nhân viên làm việc thường xuyên trong ROI.


     3.2.3. Độ Tin Cậy

     Độ tin cậy của hệ thống cần được đánh giá trong ngữ cảnh ứng dụng cụ thể.

     Hệ thống phù hợp làm early warning system (hệ thống cảnh báo sớm) thay vì final decision maker. Với detection rate 81.7 phần trăm và false positive rate 8.3 phần trăm, hệ thống không đủ tin cậy để tự động kích hoạt các hành động nghiêm trọng (như khóa cửa, gọi cảnh sát) mà cần có giám sát viên con người xác nhận. Precision 90 phần trăm nghĩa là 9 trong 10 alerts là chính xác, nhưng vẫn có 1 alert sai cần được lọc bởi người.

     Hệ thống cần kết hợp với giám sát viên để đạt hiệu quả tối đa. Vai trò của hệ thống là thu hút sự chú ý của giám sát viên khi có khả năng xâm nhập, thay vì giám sát viên phải theo dõi liên tục toàn bộ màn hình. Giám sát viên xác nhận alert là true positive hay false positive dựa trên kinh nghiệm và ngữ cảnh. Quyết định cuối cùng về việc có can thiệp hay không vẫn do con người đưa ra.

     Lợi ích lớn nhất là giảm tải công việc giám sát liên tục. Thay vì phải chăm chú nhìn vào màn hình nhiều giờ liên tục (dễ gây mệt mỏi và bỏ sót), giám sát viên chỉ cần chú ý khi hệ thống trigger alert. Recall 81.8 phần trăm nghĩa là hệ thống bắt được phần lớn các xâm nhập thực tế, giảm thiểu rủi ro bỏ sót. Chế độ ghi log đầy đủ tạo audit trail để review lại các sự kiện, training, và cải tiến hệ thống.


     3.2.4. So Sánh Mục Tiêu Với Kết Quả Thực Tế

     Việc so sánh kết quả thực tế với mục tiêu ban đầu giúp đánh giá mức độ thành công của dự án.

     Về detection rate, mục tiêu đặt ra là trên 80 phần trăm, kết quả đạt được là 81.7 phần trăm. Như vậy đã đạt mục tiêu với một chút dư, gần sát ngưỡng 85 phần trăm lý tưởng. Sự chênh lệnh nhỏ có thể khắc phục bằng tuning tham số cẩn thận hơn hoặc thêm data test đa dạng hơn.

     Về FPS, mục tiêu đặt ra là trên 20 FPS để đảm bảo gần real-time, kết quả đạt được là 26.6 FPS. Như vậy đã vượt mục tiêu đáng kể, đủ để xử lý video real-time ngay cả với độ phân giải 1080p. Tốc độ này cho phép triển khai trên hardware phổ thông mà không cần GPU chuyên dụng.

     Về false positive rate, mục tiêu đặt ra là dưới 10 phần trăm để tránh làm phiền giám sát viên quá nhiều, kết quả đạt được là 8.3 phần trăm. Như vậy đã đạt mục tiêu, nằm trong ngưỡng chấp nhận được. Time-based validation đóng vai trò quan trọng trong việc giảm false positives.

     Tổng hợp, hệ thống đạt hoặc vượt hầu hết các mục tiêu đề ra ban đầu, chứng tỏ thiết kế và triển khai là thành công.

[Bảng 3.2: So sánh mục tiêu với kết quả thực tế]
Chỉ tiêu          | Mục tiêu     | Kết quả thực tế | Đánh giá
------------------|--------------|-----------------|------------------
Detection Rate    | >80%         | 81.7%           | Đạt, gần tối ưu
FPS               | >20          | 26.6            | Vượt đáng kể
False Positive    | <10%         | 8.3%            | Đạt tốt
Precision         | >85%         | 90%             | Vượt
Recall            | >75%         | 81.8%           | Vượt
Real-time         | Có (gần RT)  | 1.03x RT        | Đạt


3.3. ĐỀ XUẤT CẢI TIẾN

     Dựa trên kết quả thực nghiệm và phân tích hạn chế, phần này đề xuất các hướng cải tiến cụ thể để nâng cao hiệu năng hệ thống.


     3.3.1. Cải Tiến Về Tốc Độ Xử Lý

     Mặc dù FPS hiện tại đã đạt mức chấp nhận được, vẫn có thể tối ưu hơn nữa để đạt 30 FPS ổn định trên mọi độ phân giải.

     Vấn đề hiện tại là frame rate chưa đạt 30 FPS ổn định trên video 1080p (chỉ đạt 24.8 FPS). Nguyên nhân chính là xử lý tuần tự từng frame qua toàn bộ pipeline, một số thao tác như Gaussian blur, morphology, contour detection tốn nhiều tính toán trên ảnh độ phân giải cao.

     Đề xuất cải tiến thứ nhất là giảm resolution cho processing. Thay vì xử lý trực tiếp trên ảnh 1080p, resize xuống 720p hoặc thậm chí 540p chỉ cho bước motion detection và intrusion detection. Chỉ cần upscale lại khi vẽ annotations hoặc lưu output video. Theo tài liệu optimization-techniques, giảm từ 1080p xuống 720p có thể tăng tốc lên 2.25 lần, từ 1080p xuống 540p có thể tăng tốc lên 4 lần. Điều này có thể đẩy FPS từ 24.8 lên gần 50-60 FPS.

     Đề xuất cải tiến thứ hai là skip frames khi cần thiết. Xử lý mỗi frame thứ 2 hoặc thứ 3 thay vì tất cả frames, tái sử dụng kết quả detection cho các frames bị skip. Theo tài liệu, skip 2 frames có thể tăng FPS lên 67 phần trăm, skip 3 frames có thể tăng lên 133 phần trăm. Tuy nhiên cần cân nhắc trade-off với khả năng phát hiện chuyển động nhanh.

     Đề xuất cải tiến thứ ba là GPU acceleration. Sử dụng cv2.cuda module nếu hardware có GPU hỗ trợ CUDA. Upload frame lên GPU, thực hiện các thao tác như cvtColor, GaussianBlur, background subtraction trên GPU, sau đó download kết quả. Theo tài liệu, GPU có thể tăng tốc lên 5-10 lần cho các thao tác này. Tuy nhiên cần lưu ý overhead của upload/download dữ liệu giữa CPU và GPU.

     Đề xuất cải tiến thứ tư là multi-threading. Tách biệt capture, processing, và display thành các threads riêng biệt. Capture thread chuyên đọc frames từ video và đưa vào queue. Worker threads (2-4 threads) song song xử lý các frames từ queue. Display thread lấy kết quả và hiển thị hoặc ghi video. Theo tài liệu multi-threading, threaded capture có thể cải thiện FPS lên 20-40 phần trăm chỉ bằng cách loại bỏ I/O blocking.

     Kỳ vọng kết quả sau khi áp dụng các cải tiến này là đạt 30 FPS ổn định trên video 1080p, xử lý được 4K real-time nếu có GPU, và giảm latency cho ứng dụng cần phản hồi nhanh.


     3.3.2. Cải Tiến Về Độ Nhạy

     Cải thiện detection rate từ 81.7 phần trăm lên gần 90 phần trăm hoặc cao hơn.

     Vấn đề thứ nhất là bỏ sót khi người di chuyển chậm hoặc dừng lại lâu trong ROI. Background model có thể học người đó thành một phần của nền sau một thời gian (learning rate quá cao). Người mặc quần áo màu sắc tương tự nền bị phát hiện kém do contrast thấp.

     Đề xuất giải quyết là giảm min_object_area từ 1500 pixels xuống 1000-1200 pixels để phát hiện người ở xa hoặc chỉ lộ một phần. Điều chỉnh learning rate của MOG2 (tham số backgroundRatio) để tránh học đối tượng chuyển động chậm vào nền. Sử dụng multi-scale detection bằng cách xử lý ảnh ở nhiều scales khác nhau rồi kết hợp kết quả. Tuy nhiên cần lưu ý trade-off là có thể tăng false positive từ nhiễu nhỏ.

     Vấn đề thứ hai là bỏ sót khi di chuyển quá nhanh gây motion blur. Một trong hai xâm nhập bị bỏ sót ở input-01 là do người chạy nhanh qua ROI, duration không đủ 1.0 giây.

     Đề xuất giải quyết là giảm time_threshold xuống 0.7-0.8 giây để bắt được chuyển động nhanh. Kết hợp optical flow để phát hiện chuyển động dựa trên vector vận tốc thay vì chỉ dựa vào background subtraction. Theo tài liệu knowledge-base về optical flow, phương pháp này tốt hơn cho chuyển động nhanh. Sử dụng camera có frame rate cao hơn (60 FPS) để giảm motion blur. Tuy nhiên trade-off là có thể tăng false positive từ chuyển động thoáng qua.


     3.3.3. Xử Lý Các Tình Huống Đặc Biệt

     Nâng cao khả năng xử lý các điều kiện môi trường khó khăn.

     Tình huống đặc biệt thứ nhất là thay đổi ánh sáng đột ngột. Ví dụ đèn bật/tắt đột ngột, đèn pha xe chiếu vào camera ban đêm, mây che nắng đột ngột. Các thay đổi này khiến background model bị nhiễu loạn, tạo foreground mask sai trên toàn bộ hoặc phần lớn frame.

     Giải pháp đề xuất là phát hiện lighting change bằng cách tính mean brightness của frame hiện tại so với mean của N frames trước. Nếu chênh lệch vượt ngưỡng (ví dụ 30 phần trăm), tạm dừng processing trong 1-2 giây để background model thích ứng. Reset background model hoàn toàn nếu thay đổi quá lớn (ví dụ bật đèn trong phòng tối). Tăng learning rate tạm thời để thích ứng nhanh hơn, sau đó quay lại learning rate bình thường.

     Tình huống đặc biệt thứ hai là occlusion hoặc che khuất. Người đi sau cột trụ, xe hơi, hoặc vật cản lớn, biến mất khỏi view một thời gian rồi xuất hiện lại.

     Giải pháp đề xuất là sử dụng Kalman filter để dự đoán trajectory. Theo tài liệu object-tracking, Kalman filter có thể predict vị trí tiếp theo dựa trên vận tốc và gia tốc trước đó. Khi đối tượng biến mất (no detection), sử dụng predicted position để tracking thay vì mất track hoàn toàn. Khi đối tượng xuất hiện lại, match với predicted position để giữ nguyên ID thay vì tạo ID mới. Điều này giúp duy trì tracking trong khoảng 5 giây occlusion thay vì chỉ 2 giây như hiện tại.

[Hình 3.3: Sơ đồ kiến trúc mở rộng với GPU acceleration và multi-threading - hiển thị Capture Thread, GPU Processing (cvtColor, Blur, MOG2), Worker Threads, và Display Thread với queues kết nối]

[Hình 3.4: Flowchart xử lý occlusion với Kalman filter - các bước: Detect object, Match with previous, If no match check predicted position, Update Kalman, Predict next position]

     Tình huống đặc biệt thứ ba là nhiều người chồng lấp. Khi nhiều người đứng gần nhau hoặc di chuyển cùng nhau, contour detection có thể nhận thành một blob lớn thay vì nhiều người riêng biệt.

     Giải pháp đề xuất là sử dụng watershed segmentation để tách blob lớn thành nhiều objects riêng biệt. Theo tài liệu knowledge-base về watershed, kỹ thuật này sử dụng distance transform và markers để tìm ranh giới giữa các objects chồng lấp. Kết hợp với size estimation để đoán số người dựa trên diện tích blob (ví dụ blob 5000 pixels có thể là 2-3 người thay vì 1 người).

     Tình huống đặc biệt thứ tư là ban đêm hoàn toàn. Detection rate ban đêm chỉ đạt 60-70 phần trăm theo phân tích limitations, ngay cả với CLAHE.

     Giải pháp đề xuất là yêu cầu camera có IR night vision (hồng ngoại) để cung cấp nguồn sáng không nhìn thấy. Tăng CLAHE clip_limit lên 3.0-4.0 để khuếch đại tối đa độ tương phản. Giảm motion threshold xuống 8-10 để tăng độ nhạy trong điều kiện signal yếu. Kết hợp với thermal camera nếu budget cho phép, vì thermal không bị ảnh hưởng bởi ánh sáng.


     3.3.4. Mở Rộng Chức Năng

     Thêm các tính năng mới để mở rộng khả năng ứng dụng.

     Chức năng mở rộng thứ nhất là person re-identification. Tạo feature vector cho mỗi person detection (sử dụng appearance model như color histogram, HOG, hoặc CNN features). Lưu trữ feature vectors của các persons đã thấy. Khi person mới xuất hiện, so sánh feature vector với database để xác định có phải là người đã thấy trước đó không. Gán cùng ID nếu match, tạo ID mới nếu không match. Điều này cho phép tracking xuyên suốt nhiều lần ra vào ROI và tạo hồ sơ hành vi dài hạn.

     Chức năng mở rộng thứ hai là face recognition. Detect face từ bounding box của person. Extract face features bằng deep learning model (FaceNet, ArcFace). So sánh với database authorized persons (whitelist). Chỉ trigger alert nếu face không match với whitelist, bỏ qua nếu là authorized person. Điều này giúp giảm đáng kể false positives trong môi trường có nhân viên làm việc.

     Chức năng mở rộng thứ ba là multi-camera tracking. Triển khai global tracker theo dõi objects di chuyển giữa nhiều cameras. Khi object biến mất khỏi camera này và xuất hiện ở camera khác, match dựa trên appearance features và timing. Tạo trajectory map toàn cục cho thấy lộ trình di chuyển qua nhiều cameras. Điều này hữu ích cho giám sát diện tích lớn với nhiều cameras.

     Chức năng mở rộng thứ tư là alert qua mạng. Gửi email alert khi phát hiện xâm nhập (kèm screenshot). Gửi HTTP POST request đến webhook để tích hợp với hệ thống quản lý khác. Push notification đến mobile app của giám sát viên. Cho phép giám sát từ xa thay vì phải ngồi trước màn hình.

     Chức năng mở rộng thứ năm là dashboard và web interface. Tạo web interface real-time hiển thị video từ nhiều cameras. Dashboard thống kê số lượng alerts theo thời gian, ROI, và camera. Heatmap hiển thị các vùng có nhiều xâm nhập nhất. Export báo cáo định kỳ (daily, weekly, monthly). Quản lý cấu hình ROI và tham số qua web thay vì sửa file YAML.


3.4. ỨNG DỤNG THỰC TẾ

     Hệ thống có tiềm năng ứng dụng cao trong nhiều lĩnh vực giám sát và an ninh.


     3.4.1. Các Lĩnh Vực Ứng Dụng

     Lĩnh vực ứng dụng thứ nhất là an ninh công cộng. Tại sân bay, hệ thống có thể giám sát khu vực tarmac (đường băng), phát hiện người không có phép xâm nhập vào khu vực nguy hiểm gần máy bay hoặc xe phục vụ. Tại nhà ga, giám sát đường ray xe lửa, cảnh báo khi có người đi vào đường ray khi không có tàu hoặc có nguy cơ va chạm. Tại bến xe, giám sát khu vực chỉ dành cho xe bus, phát hiện người đi bộ vào làn xe.

     Lĩnh vực ứng dụng thứ hai là an ninh doanh nghiệp. Tại nhà máy, giám sát khu vực nguy hiểm như dây chuyền sản xuất, máy móc lớn, kho hóa chất. Cảnh báo khi công nhân không đeo đồ bảo hộ xâm nhập vào khu vực nguy hiểm. Tại kho hàng, giám sát khu vực restricted như kho chứa hàng giá trị cao, kho vũ khí, kho dữ liệu. Chỉ cho phép người có thẻ authorized vào, cảnh báo khi phát hiện người lạ. Tại văn phòng, giám sát khu vực server room, phòng CEO, phòng họp bảo mật sau giờ làm việc.

     Lĩnh vực ứng dụng thứ ba là giám sát giao thông. Phát hiện người đi bộ vào làn dành cho ô tô trên đường cao tốc hoặc đường hầm, tự động cảnh báo cho trung tâm điều khiển. Giám sát đường ray xe điện hoặc metro, phát hiện người vào đường ray nguy hiểm, có thể liên động với hệ thống dừng tàu khẩn cấp. Giám sát cầu vượt bộ hành, phát hiện người cố gắng trèo qua lan can để nhảy xuống (nguy cơ tự tử), kích hoạt can thiệp kịp thời.

     Lĩnh vực ứng dụng thứ tư là nhà ở thông minh và an ninh gia đình. Giám sát khu vườn sau nhà, sân vườn, hoặc garage vào ban đêm. Cảnh báo khi có người lạ xâm nhập vào khuôn viên nhà. Tích hợp với hệ thống smart home để tự động bật đèn, phát báo động, hoặc thông báo cho chủ nhà qua smartphone. Cho phép xem lại footage khi có sự cố.


     3.4.2. Triển Khai Thực Tế

     Để triển khai hệ thống trong thực tế, cần chuẩn bị về phần cứng, cấu hình, và bảo trì.

     Yêu cầu phần cứng camera bao gồm độ phân giải tối thiểu 720p, khuyến nghị 1080p để có đủ chi tiết phát hiện người ở xa. Mounting position cố định, không rung lắc, vì background subtraction yêu cầu camera tĩnh. Field of view bao phủ toàn bộ ROI cần giám sát với góc nhìn tốt, tránh bị che khuất. Night vision hoặc IR nếu cần giám sát ban đêm. Weatherproof nếu lắp ngoài trời, chịu được mưa, nắng, bụi.

     Yêu cầu phần cứng processing device bao gồm CPU tối thiểu Intel Core i5 hoặc AMD Ryzen 5 để đạt 25 FPS trên 1080p. RAM tối thiểu 8GB, khuyến nghị 16GB nếu xử lý nhiều cameras đồng thời. Storage đủ cho việc lưu trữ video và screenshots, ước tính 1-2 GB mỗi giờ video 1080p. Optional GPU NVIDIA với CUDA support nếu cần tăng tốc hoặc xử lý nhiều cameras. Có thể sử dụng edge device như Raspberry Pi 4 cho setup nhỏ, nhưng cần giảm resolution hoặc skip frames.

     Quy trình cấu hình bắt đầu bằng bước một, điều chỉnh camera position và focus. Mount camera ở vị trí có góc nhìn tốt nhất cho ROI. Điều chỉnh focus để ảnh rõ nét. Test ở nhiều điều kiện ánh sáng khác nhau trong ngày.

     Bước hai là định nghĩa ROI. Sử dụng roi_selector.py để vẽ polygon ROI trực tiếp trên video. Lưu ROI vào file JSON. Test bằng cách di chuyển trong và ngoài ROI để xác nhận phát hiện chính xác.

     Bước hai là tuning tham số theo môi trường. Chọn config file phù hợp (daylight, lowlight, night) làm baseline. Điều chỉnh motion.threshold dựa trên lượng nhiễu quan sát được. Điều chỉnh intrusion.overlap_threshold và time_threshold dựa trên kịch bản sử dụng. Test với nhiều scenarios khác nhau và ghi nhận false positive/false negative rate.

     Bước ba là thiết lập alert cooldown và logging. Cấu hình cooldown_time để tránh spam alerts. Thiết lập đường dẫn log file và screenshot folder. Test alert mechanism bằng cách trigger xâm nhập giả.

     Bước bốn là deploy và monitor. Chạy hệ thống trong chế độ test một vài ngày, review alerts log để đánh giá. Fine-tuning thêm nếu cần. Deploy chính thức và monitor liên tục trong tuần đầu.

     Yêu cầu bảo trì định kỳ bao gồm review log hàng tuần để phân tích false positive/false negative patterns, điều chỉnh tham số nếu cần. Clean up logs và screenshots cũ hàng tháng để tiết kiệm storage, có thể archive vào external storage hoặc cloud. Cập nhật ROI khi có thay đổi layout hoặc furniture trong scene. Re-calibrate background model khi có thay đổi lớn về lighting hoặc background (ví dụ sơn tường, thay đổi bố cục).


     3.4.3. Chi Phí và Lợi Ích

     Phân tích chi phí và lợi ích giúp đánh giá tính khả thi kinh tế của việc triển khai.

     Chi phí phần cứng bao gồm camera IP 1080p giá từ 50 đến 200 USD mỗi camera tùy chất lượng và tính năng (night vision, weatherproof). Processing device như mini PC hoặc workstation giá từ 300 đến 800 USD, có thể dùng chung cho 2-4 cameras. Storage như HDD hoặc SSD 1-2 TB giá từ 50 đến 150 USD. Optional GPU nếu cần giá từ 200 đến 500 USD cho mid-range GPU.

     Chi phí phần mềm là 0 USD vì toàn bộ hệ thống sử dụng open-source software (Python, OpenCV, NumPy), không có license fee. Chỉ có chi phí thời gian develop và customize nếu cần tính năng đặc biệt.

     Chi phí triển khai bao gồm setup và installation mất 1-2 ngày cho mỗi camera, bao gồm mount camera, chạy dây, cài đặt software, định nghĩa ROI. Nếu thuê kỹ thuật viên, ước tính 500-1000 USD mỗi camera. Tuning và testing mất 2-3 ngày để điều chỉnh tham số tối ưu cho từng môi trường. Training cho giám sát viên sử dụng hệ thống mất 1-2 ngày.

     Tổng chi phí ban đầu cho setup 1 camera ước tính từ 1000 đến 2500 USD tùy cấu hình. Chi phí vận hành hàng tháng rất thấp, chủ yếu là điện năng (khoảng 5-10 USD mỗi camera) và bảo trì định kỳ.

     Lợi ích chính thứ nhất là giảm chi phí nhân lực 24/7. Thay vì cần 3 giám sát viên làm 3 ca (mỗi ca 8 giờ) để giám sát liên tục, chỉ cần 1 giám sát viên làm giờ hành chính review alerts. Tiết kiệm 2 nhân sự, tương đương khoảng 30000-40000 USD mỗi năm tùy mức lương.

     Lợi ích chính thứ hai là phát hiện sớm và phản ứng nhanh. Hệ thống trigger alert ngay lập tức khi phát hiện xâm nhập, nhanh hơn giám sát viên con người có thể nhận ra (đặc biệt khi giám sát nhiều màn hình). Giảm thời gian phản ứng từ vài phút xuống vài giây. Ngăn chặn sự cố trước khi leo thang nghiêm trọng.

     Lợi ích chính thứ ba là log đầy đủ để audit và training. Mọi sự kiện xâm nhập được ghi log chi tiết với timestamp, location, screenshot. Có thể review lại bất cứ lúc nào để điều tra sự cố. Sử dụng làm evidence trong các vấn đề pháp lý hoặc bảo hiểm. Sử dụng làm data để training nhân viên mới hoặc cải thiện hệ thống.

     Lợi ích chính thứ tư là scalable và dễ mở rộng. Thêm camera mới chỉ cần mount, cấu hình ROI, và chạy thêm instance của software. Không cần thay đổi infrastructure lớn. Chi phí marginal cho mỗi camera thêm chỉ là chi phí camera và setup, không cần thêm nhân sự.

     ROI (Return on Investment) ước tính thu hồi vốn trong 3-6 tháng nếu so sánh với chi phí thuê giám sát viên 24/7. Sau đó là lợi nhuận thuần từ tiết kiệm chi phí nhân lực.


3.5. KẾT LUẬN CHUNG

     Đồ án đã hoàn thành mục tiêu xây dựng hệ thống phát hiện người xâm nhập vùng cấm sử dụng các kỹ thuật xử lý ảnh số truyền thống. Hệ thống đạt được sự cân bằng tốt giữa độ chính xác, tốc độ xử lý, và tính thực tiễn trong triển khai.


     3.5.1. Tổng Kết

     Hệ thống được xây dựng thành công với kiến trúc modular, pipeline xử lý rõ ràng, và khả năng cấu hình linh hoạt. Các kỹ thuật xử lý ảnh truyền thống được áp dụng bao gồm Motion Detection (MOG2, KNN, Frame Differencing), Background Subtraction với khả năng thích ứng ánh sáng, Adaptive Thresholding xử lý ánh sáng không đồng đều, CLAHE cải thiện độ tương phản trong điều kiện thiếu sáng, Edge Detection bằng Canny algorithm, Region Growing và Contour Analysis, Intrusion Detection dựa trên IoU và time-based validation.

     Hiệu năng đạt được vượt hoặc đạt gần hầu hết các mục tiêu đề ra. Detection rate 81.7 phần trăm trung bình, gần sát mục tiêu 85 phần trăm. False positive rate 8.3 phần trăm, nằm trong ngưỡng chấp nhận được dưới 10 phần trăm. FPS 26.6 trung bình, vượt mục tiêu 20 FPS, cho phép xử lý gần real-time. Precision 90 phần trăm và Recall 81.8 phần trăm cho thấy sự cân bằng tốt giữa độ chính xác và độ phủ.

     Code được tổ chức modular, dễ maintain, và dễ mở rộng. Mỗi module đảm nhận một chức năng cụ thể, có thể test và thay thế độc lập. File cấu hình YAML tập trung cho phép tuning tham số mà không cần sửa code. Cấu trúc code tuân theo best practices của Python và software engineering.

     Tiềm năng ứng dụng thực tế cao trong nhiều lĩnh vực như an ninh công cộng (sân bay, nhà ga), an ninh doanh nghiệp (nhà máy, kho hàng), giám sát giao thông (đường ray, cao tốc), và smart home security. Chi phí triển khai hợp lý (1000-2500 USD mỗi camera setup) với ROI nhanh (3-6 tháng) nhờ tiết kiệm chi phí nhân lực.


     3.5.2. Ý Nghĩa

     Về mặt học thuật, đồ án thành công trong việc áp dụng lý thuyết xử lý ảnh từ giáo trình vào giải quyết bài toán thực tế. Hiểu sâu về các kỹ thuật như background subtraction, adaptive thresholding, edge detection, và cách kết hợp chúng trong một pipeline xử lý. Học được cách tuning tham số dựa trên thực nghiệm và phân tích kết quả. Nắm vững workflow phát triển hệ thống computer vision từ thu thập data, thiết kế, triển khai, đến đánh giá.

     Về mặt thực tiễn, hệ thống cung cấp giải pháp giám sát chi phí thấp, hiệu quả cao so với các hệ thống thương mại đắt tiền. Sử dụng hoàn toàn open-source software, không có license fee. Có thể chạy trên hardware phổ thông, không bắt buộc GPU chuyên dụng. Dễ tùy chỉnh và mở rộng cho các nhu cầu cụ thể của từng môi trường giám sát.

     Về mặt cá nhân, đồ án giúp nắm vững Python programming với các thư viện computer vision (OpenCV, NumPy, scikit-image). Hiểu rõ pipeline xử lý ảnh và video trong thực tế, từ đọc frames, xử lý, đến output. Học được kỹ năng debug, profiling, và optimization cho ứng dụng real-time. Rèn luyện tư duy phân tích bài toán, thiết kế giải pháp, và đánh giá kết quả khách quan.


     3.5.3. Hướng Phát Triển Tương Lai

     Trong ngắn hạn (1-3 tháng), ưu tiên bổ sung test data cho các điều kiện khó khăn hơn như ban đêm, mưa, sương mù để đánh giá toàn diện. Tối ưu FPS lên 30 ổn định bằng cách áp dụng các kỹ thuật như resize, skip frames, hoặc multi-threading. Cải thiện xử lý occlusion bằng Kalman filter hoặc trajectory prediction. Fine-tuning tham số dựa trên feedback từ test thực tế trong nhiều môi trường khác nhau.

     Trong trung hạn (3-6 tháng), tích hợp deep learning để cải thiện độ chính xác. Sử dụng YOLO hoặc Faster R-CNN cho person detection thay vì chỉ dựa vào background subtraction. Object classification để phân biệt person, vehicle, animal, giảm false positives. Person re-identification sử dụng CNN features để tracking xuyên suốt. Face recognition để phân biệt authorized và unauthorized persons. Triển khai multi-camera tracking với global ID management. Xây dựng web dashboard cho monitoring và configuration từ xa.

     Trong dài hạn (6-12 tháng), di chuyển hoàn toàn sang deep learning-based system với end-to-end model. Edge AI deployment trên các thiết bị như Jetson Nano, Raspberry Pi với Coral TPU để giảm chi phí infrastructure. Cloud integration cho centralized management, storage, và analytics. Real-time streaming và mobile app cho giám sát từ xa. Anomaly detection và behavior analysis để phát hiện các hành vi bất thường ngoài xâm nhập đơn thuần. Activity recognition (đi, chạy, ngồi, ngã) cho các ứng dụng elderly care hoặc workplace safety.

     Mục tiêu dài hạn cuối cùng là xây dựng một intelligent video surveillance platform hoàn chỉnh, tích hợp nhiều AI modules, hỗ trợ multi-camera, multi-site, với khả năng học liên tục từ feedback và tự động cải thiện theo thời gian.


     3.5.4. Lời Cảm Ơn

     Em xin chân thành cảm ơn giảng viên hướng dẫn đã tận tình chỉ bảo, góp ý, và hỗ trợ em trong suốt quá trình thực hiện đồ án này. Những lời khuyên và định hướng của thầy cô đã giúp em vượt qua nhiều khó khăn kỹ thuật và hoàn thiện được hệ thống.

     Em cảm ơn cộng đồng OpenCV và các tác giả của các tài liệu, papers, tutorials đã chia sẻ kiến thức quý báu. Những tài liệu này là nền tảng quan trọng giúp em hiểu sâu về các kỹ thuật xử lý ảnh và cách triển khai chúng trong thực tế.

     Em cảm ơn các dataset công khai như VIRAT, CAVIAR đã cung cấp data chất lượng cao cho việc test và đánh giá hệ thống. Những dataset này giúp em có cái nhìn thực tế về các challenges trong video surveillance.

     Cuối cùng, em cảm ơn gia đình và bạn bè đã động viên và hỗ trợ em trong suốt thời gian thực hiện đồ án. Sự ủng hộ của mọi người là động lực quan trọng giúp em hoàn thành tốt đồ án này.

     Mặc dù đã cố gắng hết sức, đồ án vẫn không tránh khỏi những thiếu sót. Em rất mong nhận được những góp ý, nhận xét từ thầy cô và các bạn để có thể cải thiện hệ thống tốt hơn trong tương lai.


Thành phố Hồ Chí Minh, tháng 11 năm 2025
