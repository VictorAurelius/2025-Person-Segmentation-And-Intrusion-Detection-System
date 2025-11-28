CHƯƠNG 1: CƠ SỞ LÝ THUYẾT


1.1. TỔNG QUAN VỀ XỬ LÝ ẢNH SỐ

     Xử lý ảnh số đóng vai trò quan trọng trong nhiều lĩnh vực khoa học và công nghệ hiện đại, đặc biệt trong các hệ thống giám sát an ninh. Ảnh số (digital image) là một hàm hai chiều f(x, y), trong đó x và y là tọa độ không gian trong mặt phẳng, còn giá trị của hàm f tại mỗi điểm biểu diễn cường độ sáng (intensity) hoặc màu sắc tại điểm đó.

     Trong biểu diễn số, một ảnh được mô tả dưới dạng ma trận các điểm ảnh (pixel). Đối với ảnh xám (grayscale), mỗi pixel có giá trị từ 0 đến 255, trong đó 0 đại diện cho màu đen hoàn toàn và 255 đại diện cho màu trắng hoàn toàn. Đối với ảnh màu, mỗi pixel được biểu diễn bởi ba giá trị trong không gian màu RGB (Red, Green, Blue) hoặc các không gian màu khác như HSV (Hue, Saturation, Value). Độ phân giải của ảnh được xác định bởi số lượng pixel theo chiều ngang và chiều dọc, ví dụ như 1920x1080 pixels (Full HD) hay 1280x720 pixels (HD).

     Trong bài toán giám sát an ninh và phát hiện xâm nhập, xử lý ảnh số đóng vai trò then chốt trong việc tự động hóa quá trình phát hiện và cảnh báo. Thay vì yêu cầu con người phải theo dõi liên tục các camera giám sát, hệ thống xử lý ảnh có thể tự động phân tích video, phát hiện chuyển động, nhận dạng đối tượng và đưa ra cảnh báo khi có xâm nhập vào khu vực cấm. Điều này giúp giảm chi phí nhân lực, tăng độ chính xác và khả năng phản ứng nhanh chóng với các tình huống an ninh.

     Tuy nhiên, việc xử lý ảnh trong môi trường giám sát thực tế phải đối mặt với nhiều thách thức. Thứ nhất, điều kiện ánh sáng thay đổi liên tục theo thời gian trong ngày, từ ánh sáng mặt trời mạnh ban ngày đến thiếu sáng hoặc tối hoàn toàn ban đêm. Thứ hai, hiện tượng che khuất (occlusion) xảy ra khi đối tượng bị che khuất một phần hoặc hoàn toàn bởi các vật thể khác. Thứ ba, nhiễu ảnh (noise) từ sensor camera, nén video, hoặc điều kiện môi trường như mưa, sương mù có thể ảnh hưởng đến chất lượng ảnh. Thứ tư, bóng đổ (shadows) và phản chiếu ánh sáng có thể gây ra các phát hiện sai (false positives).

     Ứng dụng thực tế của xử lý ảnh trong giám sát an ninh rất đa dạng. Tại sân bay và nhà ga, hệ thống có thể phát hiện người xâm nhập vào khu vực hạn chế như đường băng hay khu vực kỹ thuật. Trong nhà máy và kho hàng, hệ thống cảnh báo khi có người vào khu vực nguy hiểm như khu vực máy móc đang hoạt động. Tại các công trình công cộng, hệ thống giám sát các khu vực cấm ra vào và tự động ghi lại bằng chứng khi có vi phạm.

[Hình 1.1: Biểu diễn ảnh số dưới dạng ma trận pixel với các giá trị cường độ sáng từ 0 đến 255]

[Hình 1.2: Các không gian màu phổ biến - RGB (Red, Green, Blue), Grayscale (ảnh xám), và HSV (Hue, Saturation, Value)]


1.2. MOTION DETECTION - PHÁT HIỆN CHUYỂN ĐỘNG

     Phát hiện chuyển động là bước đầu tiên và quan trọng nhất trong hệ thống giám sát, giúp xác định vùng có hoạt động trong video để tiết kiệm tài nguyên xử lý và tập trung vào các đối tượng quan tâm. Có hai phương pháp chính được sử dụng: Frame Differencing (so sánh khung hình) và Background Subtraction (trừ nền).

     Frame Differencing là phương pháp đơn giản nhất, hoạt động bằng cách so sánh sự khác biệt pixel giữa các khung hình liên tiếp. Nguyên lý cơ bản được biểu diễn bằng công thức: Chuyển động = |Frame(t) - Frame(t-1)|, trong đó Frame(t) là khung hình hiện tại và Frame(t-1) là khung hình trước đó. Nếu sự khác biệt giữa hai khung hình vượt quá một ngưỡng nhất định, hệ thống coi đó là chuyển động.

     Ưu điểm chính của Frame Differencing là tốc độ xử lý rất nhanh, phù hợp với các hệ thống yêu cầu real-time. Phương pháp này không cần thời gian học (learning phase) và có thể hoạt động ngay lập tức. Tuy nhiên, phương pháp này có nhược điểm là nhạy cảm với nhiễu ảnh, dễ bị ảnh hưởng bởi thay đổi ánh sáng đột ngột, và không xử lý tốt các đối tượng di chuyển chậm. Khi đối tượng di chuyển quá chậm, sự khác biệt giữa hai khung hình liên tiếp có thể quá nhỏ, không đủ để vượt qua ngưỡng phát hiện.

     Background Subtraction là phương pháp tiên tiến hơn, hoạt động bằng cách xây dựng một mô hình nền (background model) và so sánh mỗi khung hình hiện tại với mô hình này để tách ra các đối tượng chuyển động (foreground). Phương pháp phổ biến nhất là MOG2 (Mixture of Gaussians 2), được phát triển bởi Zivkovic năm 2004.

     MOG2 mô hình hóa mỗi pixel bằng một hỗn hợp các phân phối Gaussian. Cụ thể, mỗi pixel I(x,y) được biểu diễn bằng tổng của K phân phối Gaussian với các trọng số khác nhau. Mô hình này có khả năng thích ứng tự động với các thay đổi ánh sáng từ từ theo thời gian. Khi một pixel mới xuất hiện, thuật toán so sánh giá trị này với các phân phối Gaussian đã học. Nếu pixel phù hợp với một trong các phân phối nền, nó được phân loại là nền (background). Ngược lại, nó được phân loại là tiền cảnh (foreground - đối tượng chuyển động).

     Các tham số quan trọng của MOG2 bao gồm: history (số khung hình dùng để học mô hình nền, mặc định là 500), varThreshold (ngưỡng phát hiện, mặc định là 16), và detectShadows (phát hiện bóng đổ). Tham số history càng lớn thì mô hình nền càng ổn định nhưng thích ứng chậm hơn với thay đổi. Tham số varThreshold càng thấp thì hệ thống càng nhạy, phát hiện được nhiều chuyển động nhưng cũng tăng tỷ lệ phát hiện sai.

     Một phương pháp tương tự khác là KNN (K-Nearest Neighbors) background subtraction. Thay vì sử dụng mô hình Gaussian, KNN duy trì một tập mẫu cho mỗi pixel và so sánh pixel hiện tại với K mẫu gần nhất. Nếu khoảng cách đến các mẫu gần nhất nhỏ hơn ngưỡng, pixel được coi là nền. KNN thường cho kết quả chính xác hơn MOG2 trong môi trường nhiễu cao, nhưng tốc độ xử lý chậm hơn và yêu cầu bộ nhớ lớn hơn.

     So sánh giữa Frame Differencing và Background Subtraction cho thấy mỗi phương pháp có ưu nhược điểm riêng. Frame Differencing có tốc độ xử lý rất nhanh, phù hợp cho hệ thống yêu cầu độ trễ thấp, nhưng độ chính xác thấp hơn và nhạy cảm với nhiễu. Background Subtraction có độ chính xác cao hơn, khả năng thích ứng với thay đổi ánh sáng tốt hơn, nhưng tốc độ xử lý chậm hơn và cần thời gian học ban đầu. Đối với bài toán giám sát an ninh, MOG2 được lựa chọn làm phương pháp chính vì cân bằng tốt giữa tốc độ và độ chính xác.

[Hình 1.3: Minh họa nguyên lý Frame Differencing - so sánh hai khung hình liên tiếp để phát hiện sự thay đổi]

[Hình 1.4: Quá trình Background Subtraction với MOG2 - mô hình nền được cập nhật liên tục và so sánh với khung hình hiện tại]

[Hình 1.5: Ví dụ foreground mask (mặt nạ tiền cảnh) thu được từ MOG2, trong đó vùng trắng là đối tượng chuyển động và vùng đen là nền tĩnh]

[Bảng 1.1: So sánh Frame Differencing và Background Subtraction]
Tiêu chí           | Frame Differencing  | Background Subtraction (MOG2)
-------------------|---------------------|------------------------------
Tốc độ xử lý       | Rất nhanh          | Trung bình
Độ chính xác       | Trung bình         | Cao
Thích ứng ánh sáng | Không              | Có (tự động)
Thời gian khởi động| Ngay lập tức       | Cần học (100-200 frames)
Xử lý chuyển động chậm | Kém            | Tốt
Nhạy cảm nhiễu     | Cao                | Trung bình
Phát hiện bóng đổ  | Không              | Có (tùy chọn)


1.3. ADAPTIVE THRESHOLDING - NGƯỠNG HÓA THÍCH ỨNG

     Ngưỡng hóa (thresholding) là kỹ thuật cơ bản trong xử lý ảnh, nhằm phân chia ảnh thành hai vùng: nền trước (foreground) và nền sau (background) dựa trên cường độ sáng của pixel. Trong ngưỡng hóa toàn cục (global thresholding), một giá trị ngưỡng duy nhất được áp dụng cho toàn bộ ảnh. Tuy nhiên, phương pháp này gặp hạn chế khi ảnh có điều kiện ánh sáng không đồng đều.

     Adaptive thresholding (ngưỡng hóa thích ứng) giải quyết vấn đề này bằng cách tính toán giá trị ngưỡng khác nhau cho từng vùng cục bộ trong ảnh. Thay vì sử dụng một ngưỡng toàn cục, thuật toán chia ảnh thành nhiều khối nhỏ và tính toán ngưỡng riêng cho mỗi khối dựa trên đặc điểm ánh sáng cục bộ của khối đó. Điều này đặc biệt hữu ích trong môi trường giám sát thực tế, nơi ánh sáng thường không đồng đều do các nguồn sáng khác nhau, bóng đổ, hoặc sự che khuất.

     Phương pháp Gaussian Adaptive Threshold là một trong những kỹ thuật phổ biến nhất. Thuật toán hoạt động bằng cách tính toán ngưỡng cho mỗi pixel dựa trên trung bình có trọng số Gaussian của vùng lân cận. Công thức tính ngưỡng tại mỗi điểm là: Ngưỡng(x, y) = Trung bình có trọng số Gaussian(vùng lân cận) - C, trong đó C là một hằng số điều chỉnh được xác định thủ công.

     Hai tham số quan trọng cần điều chỉnh trong adaptive thresholding là block_size (kích thước khối) và constant C. Tham số block_size xác định kích thước vùng lân cận sử dụng để tính toán ngưỡng, và phải là số lẻ. Giá trị block_size nhỏ (như 7-11) phù hợp với ảnh có nhiều chi tiết nhỏ hoặc ánh sáng thay đổi nhanh trong phạm vi hẹp. Giá trị block_size lớn (như 21-31) phù hợp với ảnh có vùng đồng nhất lớn hoặc ánh sáng thay đổi chậm.

     Tham số C điều chỉnh độ nhạy của ngưỡng. Giá trị C nhỏ (0-2) làm cho vùng tiền cảnh mở rộng ra, phát hiện nhiều chi tiết hơn nhưng có thể tăng nhiễu. Giá trị C lớn (5-10) làm cho vùng tiền cảnh co lại, giảm nhiễu nhưng có thể mất một số chi tiết. Giá trị khuyến nghị cho C thường nằm trong khoảng 2-5 tùy thuộc vào ứng dụng cụ thể.

     Trong hệ thống phát hiện xâm nhập, adaptive thresholding được áp dụng để xử lý các trường hợp ánh sáng không đồng đều, ví dụ như khu vực có một phần được chiếu sáng bởi đèn trong khi phần khác ở trong bóng tối. Phương pháp này giúp tách đối tượng người khỏi nền một cách hiệu quả hơn so với ngưỡng hóa toàn cục.

     Một kỹ thuật bổ sung thường được sử dụng kết hợp với adaptive thresholding là CLAHE (Contrast Limited Adaptive Histogram Equalization - cân bằng histogram thích ứng có giới hạn độ tương phản). CLAHE cải thiện độ tương phản của ảnh bằng cách cân bằng histogram cục bộ, đồng thời giới hạn mức độ khuếch đại để tránh tăng nhiễu quá mức. Kỹ thuật này đặc biệt hữu ích khi xử lý video trong điều kiện thiếu sáng hoặc ban đêm.

     CLAHE hoạt động bằng cách chia ảnh thành các ô lưới (tiles) và thực hiện cân bằng histogram riêng cho mỗi ô. Hai tham số quan trọng là clipLimit (giới hạn cắt xén, thường từ 2.0 đến 4.0) và tileGridSize (kích thước lưới, thường là 8x8 hoặc 16x16). Giá trị clipLimit cao hơn tạo ra độ tương phản mạnh hơn nhưng có thể tăng nhiễu. Kích thước lưới lớn hơn làm giảm hiệu ứng cục bộ nhưng tăng tính đồng nhất.

[Hình 1.6: So sánh Global Thresholding và Adaptive Thresholding trên cùng một ảnh có ánh sáng không đồng đều]

[Hình 1.7: Kết quả áp dụng CLAHE trên ảnh thiếu sáng - cải thiện đáng kể độ tương phản và khả năng nhìn thấy chi tiết]


1.4. EDGE DETECTION - PHÁT HIỆN BIÊN

     Biên (edge) trong ảnh là những điểm có sự thay đổi đột ngột về cường độ sáng, thường tương ứng với ranh giới giữa các đối tượng hoặc giữa đối tượng và nền. Phát hiện biên là một kỹ thuật quan trọng trong thị giác máy tính, giúp xác định đường viền của đối tượng và hỗ trợ cho các bước phân vùng và nhận dạng tiếp theo.

     Gradient là khái niệm cốt lõi trong phát hiện biên, đại diện cho tốc độ thay đổi cường độ sáng trong ảnh. Tại các vị trí có biên, gradient có giá trị lớn. Các toán tử gradient cơ bản bao gồm Sobel, Prewitt, và Scharr. Sobel operator sử dụng hai ma trận nhân chập 3x3 để tính gradient theo hướng ngang (Gx) và hướng dọc (Gy). Độ lớn gradient được tính bằng công thức: Magnitude = căn bậc hai của (Gx bình phương + Gy bình phương).

     Trong số các thuật toán phát hiện biên, Canny Edge Detection được coi là một trong những phương pháp tốt nhất và được sử dụng rộng rãi nhất. Thuật toán Canny, được phát triển bởi John Canny năm 1986, hoạt động qua năm bước chính.

     Bước một là làm mịn ảnh bằng bộ lọc Gaussian (Gaussian smoothing). Mục đích của bước này là giảm nhiễu trong ảnh trước khi tính gradient, vì gradient rất nhạy cảm với nhiễu. Một ảnh nhiễu sẽ tạo ra nhiều biên giả, do đó việc làm mịn là cần thiết. Bộ lọc Gaussian là lựa chọn tối ưu vì nó làm mịn đồng thời với việc giữ lại thông tin biên.

     Bước hai là tính gradient (gradient calculation) sử dụng toán tử Sobel hoặc Scharr. Thuật toán tính gradient theo cả hai hướng ngang và dọc, sau đó tính độ lớn và hướng của gradient tại mỗi pixel. Độ lớn gradient cho biết cường độ của biên, trong khi hướng gradient cho biết biên đó nằm theo hướng nào.

     Bước ba là non-maximum suppression (triệt phi cực đại), nhằm làm mỏng các biên. Sau khi tính gradient, các biên thường có độ rộng nhiều pixel. Non-maximum suppression loại bỏ các pixel không phải là cực đại cục bộ theo hướng gradient, chỉ giữ lại các pixel nằm trên đỉnh của biên. Kết quả là các biên có độ rộng chỉ một pixel.

     Bước bốn là double thresholding (ngưỡng kép), phân loại các pixel biên thành ba nhóm: biên mạnh (strong edge), biên yếu (weak edge), và không phải biên (non-edge). Thuật toán sử dụng hai ngưỡng: ngưỡng cao (high threshold) và ngưỡng thấp (low threshold). Pixel có gradient lớn hơn ngưỡng cao được coi là biên mạnh. Pixel có gradient nằm giữa hai ngưỡng được coi là biên yếu. Pixel có gradient nhỏ hơn ngưỡng thấp bị loại bỏ.

     Bước năm là edge tracking by hysteresis (theo dõi biên bằng trễ), kết nối các biên thành đường liên tục. Tất cả các biên mạnh được giữ lại. Các biên yếu chỉ được giữ lại nếu chúng kết nối với một biên mạnh. Quá trình này giúp loại bỏ các biên yếu do nhiễu gây ra, đồng thời giữ lại các biên yếu là phần của đường viền thực sự.

     Tỷ lệ khuyến nghị giữa ngưỡng thấp và ngưỡng cao là 1:2 hoặc 1:3. Ví dụ, nếu ngưỡng thấp là 50 thì ngưỡng cao nên là 100 hoặc 150. Điều chỉnh các ngưỡng này ảnh hưởng trực tiếp đến số lượng biên được phát hiện: ngưỡng thấp phát hiện nhiều biên hơn nhưng có nhiều nhiễu, ngưỡng cao chỉ phát hiện các biên mạnh nhất.

     So sánh với Sobel operator đơn thuần, Canny cho kết quả chính xác hơn, ít nhiễu hơn, và biên có độ rộng một pixel rõ ràng hơn. Tuy nhiên, Canny phức tạp và chậm hơn Sobel do phải thực hiện nhiều bước xử lý.

     Trong hệ thống phát hiện xâm nhập, edge detection đóng vai trò hỗ trợ cho việc xác định chính xác đường viền của người. Kết quả phát hiện biên được kết hợp với motion mask để cải thiện độ chính xác của việc tách đối tượng khỏi nền. Thông tin biên cũng hữu ích cho thuật toán region growing, giúp xác định ranh giới của vùng cần mở rộng.

[Hình 1.8: Minh họa năm bước của thuật toán Canny Edge Detection - từ ảnh gốc qua làm mịn, tính gradient, triệt phi cực đại, ngưỡng kép, đến kết quả biên cuối cùng]

[Hình 1.9: So sánh kết quả phát hiện biên giữa Canny và Sobel - Canny cho biên mỏng và rõ nét hơn]


1.5. REGION GROWING - MỞ RỘNG VÙNG

     Region growing (mở rộng vùng) là một thuật toán phân vùng ảnh hoạt động bằng cách nhóm các pixel tương đồng thành các vùng liên thông. Thuật toán bắt đầu từ một hoặc nhiều điểm giống (seed points) và mở rộng dần vùng bằng cách thêm các pixel lân cận thỏa mãn tiêu chí tương đồng.

     Quy trình cơ bản của region growing gồm năm bước. Bước một là chọn điểm giống ban đầu, có thể là một điểm được chỉ định thủ công hoặc được xác định tự động dựa trên một số tiêu chí như pixel có giá trị đặc trưng hoặc tâm của một vùng chuyển động. Bước hai là kiểm tra các pixel lân cận của điểm giống. Bước ba là áp dụng tiêu chí tương đồng để quyết định pixel lân cận có được thêm vào vùng hay không. Nếu pixel lân cận đủ tương đồng với vùng hiện tại, nó được thêm vào vùng. Bước bốn là lặp lại quá trình kiểm tra và thêm pixel cho tất cả các pixel mới được thêm vào vùng. Bước năm là dừng khi không còn pixel lân cận nào thỏa mãn tiêu chí tương đồng.

     Tiêu chí tương đồng (similarity criteria) là yếu tố quan trọng quyết định cách vùng được mở rộng. Tiêu chí đơn giản nhất dựa trên cường độ sáng: một pixel được coi là tương đồng nếu sự khác biệt cường độ sáng giữa nó và điểm giống nhỏ hơn một ngưỡng cho trước. Công thức: |pixel - seed_intensity| <= threshold. Tiêu chí này phù hợp với ảnh xám hoặc khi chỉ quan tâm đến độ sáng.

     Đối với ảnh màu, tiêu chí có thể dựa trên khoảng cách màu sắc trong không gian RGB hoặc HSV. Khoảng cách Euclidean giữa vectơ màu của pixel và màu trung bình của vùng được so sánh với ngưỡng. Một tiêu chí phức tạp hơn có thể kết hợp nhiều yếu tố như cường độ sáng, gradient, và vị trí không gian với các trọng số khác nhau.

     Connectivity (kết nối) xác định các pixel nào được coi là lân cận. Có hai loại kết nối phổ biến. Kết nối 4 hướng (4-connectivity) chỉ xem xét bốn pixel ở các hướng trên, dưới, trái, phải. Kết nối 8 hướng (8-connectivity) bao gồm cả bốn pixel góc chéo, tổng cộng tám pixel lân cận. Kết nối 8 hướng tạo ra vùng liên tục hơn nhưng có thể kết nối các vùng mỏng qua góc.

     Trong bài toán phân vùng người, region growing thường được áp dụng kết hợp với kết quả từ motion detection và edge detection. Điểm giống được chọn tự động từ các vùng có chuyển động (motion mask). Edge information giúp xác định ranh giới của vùng, ngăn vùng mở rộng vượt qua các biên rõ ràng. Kết hợp này giúp tách người khỏi nền phức tạp một cách hiệu quả, ngay cả khi có sự tương đồng màu sắc giữa người và một phần của nền.

     Ưu điểm của region growing là đơn giản, trực quan, và có thể phân vùng các vùng có hình dạng phức tạp. Thuật toán cũng cho phép tích hợp nhiều tiêu chí tương đồng khác nhau. Tuy nhiên, thuật toán nhạy cảm với nhiễu và việc chọn điểm giống ban đầu. Ngoài ra, việc chọn ngưỡng tương đồng phù hợp đòi hỏi thử nghiệm và điều chỉnh.

[Hình 1.10: Quá trình Region Growing từ một seed point - vùng được mở rộng dần ra bằng cách thêm các pixel lân cận thỏa mãn tiêu chí tương đồng]


1.6. INTRUSION DETECTION - PHÁT HIỆN XÂM NHẬP

     Phát hiện xâm nhập là mục tiêu cuối cùng của hệ thống, xác định khi có đối tượng (người) xâm nhập vào khu vực cấm và kích hoạt cảnh báo. Để thực hiện điều này, cần định nghĩa vùng quan tâm (ROI - Region of Interest) và thiết lập các tiêu chí xác thực xâm nhập.

     ROI là các vùng được định nghĩa trước trong khung hình, nơi hệ thống sẽ giám sát và phát hiện xâm nhập. Có hai loại ROI phổ biến. ROI hình chữ nhật (rectangle ROI) được định nghĩa bởi tọa độ góc trên bên trái và kích thước chiều rộng, chiều cao. Loại này đơn giản và tính toán nhanh nhưng không linh hoạt về hình dạng. ROI đa giác (polygon ROI) được định nghĩa bởi một danh sách các điểm đỉnh. Loại này linh hoạt, có thể mô tả các khu vực có hình dạng bất kỳ, phù hợp với môi trường thực tế nơi các khu vực cấm thường không có hình chữ nhật đơn giản.

     Thông tin ROI thường được lưu trữ dưới dạng JSON để dễ dàng chỉnh sửa và quản lý. Mỗi ROI có các thuộc tính như tên, loại (rectangle hoặc polygon), danh sách điểm định nghĩa, và màu sắc hiển thị. Hệ thống cho phép định nghĩa nhiều ROI khác nhau trong cùng một khung hình, mỗi ROI có thể có các tham số giám sát riêng.

     Overlap detection (phát hiện chồng lấn) là bước kiểm tra xem đối tượng có xâm nhập vào ROI hay không. Có hai phương pháp chính. Phương pháp IoU (Intersection over Union - giao trên hợp) tính tỷ lệ giữa diện tích giao và diện tích hợp của bounding box đối tượng và ROI. Công thức: IoU = Diện tích giao / Diện tích hợp. Giá trị IoU nằm trong khoảng 0 đến 1, trong đó 0 nghĩa là không có chồng lấn và 1 nghĩa là chồng lấn hoàn toàn.

     Phương pháp overlap percentage (tỷ lệ chồng lấn) tính tỷ lệ phần trăm diện tích đối tượng nằm trong ROI. Công thức: Overlap = Diện tích giao / Diện tích đối tượng. Phương pháp này hữu ích khi muốn biết bao nhiêu phần trăm đối tượng đã xâm nhập vào khu vực cấm, không phụ thuộc vào kích thước của ROI.

     Để tránh phát hiện sai (false positives), hệ thống sử dụng ba tiêu chí xác thực. Tiêu chí overlap threshold (ngưỡng chồng lấn) yêu cầu tỷ lệ chồng lấn phải vượt qua một ngưỡng nhất định. Giá trị thường dùng là 0.3, nghĩa là ít nhất 30 phần trăm đối tượng phải nằm trong ROI. Ngưỡng thấp (0.2-0.3) cho cảnh báo sớm nhưng có nhiều false positives. Ngưỡng cao (0.6-0.8) đảm bảo xâm nhập thực sự nhưng có thể phản ứng chậm.

     Tiêu chí time threshold (ngưỡng thời gian) yêu cầu đối tượng phải ở trong ROI trong một khoảng thời gian tối thiểu trước khi kích hoạt cảnh báo. Giá trị thường dùng là 1.0 giây. Tiêu chí này giúp loại bỏ các phát hiện sai do người đi qua nhanh, chim bay qua, hoặc bóng đổ tạm thời.

     Tiêu chí size filter (lọc kích thước) loại bỏ các đối tượng quá nhỏ, có thể là nhiễu, bóng đổ nhỏ, hoặc các artifacts trong ảnh. Diện tích tối thiểu thường được đặt ở khoảng 1000 pixels, tùy thuộc vào độ phân giải video và kích thước dự kiến của đối tượng quan tâm.

     Temporal consistency (tính nhất quán theo thời gian) được đảm bảo thông qua cơ chế tracking (theo dõi) đối tượng qua nhiều khung hình. Hệ thống duy trì một cấu trúc dữ liệu lưu trữ thông tin về mỗi đối tượng đang được theo dõi, bao gồm thời điểm đầu tiên xuất hiện trong ROI, thời điểm xuất hiện cuối cùng, tên ROI, và tổng thời gian ở trong ROI. Khi tính toán cho mỗi khung hình mới, hệ thống cập nhật thông tin này và chỉ kích hoạt cảnh báo khi thời gian trong ROI vượt qua ngưỡng.

     Point-in-polygon test (kiểm tra điểm trong đa giác) là một thuật toán quan trọng để xác định xem một điểm có nằm trong ROI đa giác hay không. Thuật toán ray casting (phóng tia) hoạt động bằng cách kẻ một tia từ điểm cần kiểm tra theo một hướng (thường là ngang) và đếm số lần tia này cắt các cạnh của đa giác. Nếu số lần cắt là lẻ, điểm nằm trong đa giác. Nếu số lần cắt là chẵn, điểm nằm ngoài đa giác.

[Hình 1.11: Ví dụ ROI polygon được vẽ trên khung hình video giám sát, định nghĩa khu vực cấm cần giám sát]

[Hình 1.12: Minh họa tính toán IoU giữa bounding box của đối tượng và ROI - vùng giao được tô màu đậm]

[Hình 1.13: Quy trình phát hiện xâm nhập hoàn chỉnh từ video input đến alert output, bao gồm các bước motion detection, object segmentation, overlap checking, time validation, và alert triggering]


1.7. CÁC YẾU TỐ ẢNH HƯỞNG ĐẾN CHẤT LƯỢNG HỆ THỐNG

     Hiệu năng của hệ thống phát hiện xâm nhập bị ảnh hưởng bởi nhiều yếu tố môi trường và kỹ thuật. Hiểu rõ các yếu tố này giúp điều chỉnh tham số phù hợp và đạt kết quả tối ưu.

     Độ phân giải (resolution) của video ảnh hưởng trực tiếp đến độ chi tiết và tốc độ xử lý. Độ phân giải cao hơn như 1080p (1920x1080 pixels) cung cấp nhiều chi tiết hơn, giúp phát hiện đối tượng chính xác hơn, đặc biệt khi đối tượng ở xa camera. Tuy nhiên, độ phân giải cao yêu cầu tài nguyên xử lý lớn hơn, dẫn đến tốc độ xử lý chậm hơn và tiêu tốn bộ nhớ nhiều hơn.

     Ngược lại, độ phân giải thấp hơn như 720p (1280x720 pixels) hoặc 480p (640x480 pixels) giảm yêu cầu tài nguyên, cho phép xử lý nhanh hơn, phù hợp cho hệ thống cần tốc độ cao hoặc thiết bị có tài nguyên hạn chế. Tuy nhiên, chi tiết ít hơn có thể làm giảm độ chính xác, đặc biệt với đối tượng nhỏ hoặc xa. Trong thực tế, cần cân bằng giữa độ phân giải và tốc độ xử lý. Một giải pháp phổ biến là giảm độ phân giải trong quá trình xử lý nhưng giữ nguyên độ phân giải gốc khi lưu video.

     Điều kiện ánh sáng là yếu tố quan trọng nhất ảnh hưởng đến chất lượng phát hiện. Trong điều kiện ban ngày với ánh sáng tự nhiên tốt, hệ thống hoạt động tối ưu với độ chính xác cao, ít nhiễu, và tương phản rõ ràng giữa đối tượng và nền. Tham số MOG2 threshold có thể đặt ở mức trung bình hoặc cao (18-25) để giảm false positives.

     Trong điều kiện thiếu sáng như buổi tối hoặc trong nhà, độ tương phản giảm, nhiễu tăng, và khó phân biệt đối tượng. Hệ thống cần tăng độ nhạy bằng cách giảm threshold (12-15) và kích hoạt CLAHE để cải thiện độ tương phản. Tuy nhiên, điều này có thể tăng false positives do nhiễu.

     Trong điều kiện ban đêm hoàn toàn, nhiễu rất cao, chi tiết hầu như mất hết, và hệ thống khó hoạt động nếu không có nguồn sáng bổ sung. Giải pháp là sử dụng camera có chế độ tầm nhìn ban đêm (infrared), đèn chiếu sáng bổ sung, hoặc camera có độ nhạy sáng cao. Ngay cả với hỗ trợ này, threshold cần đặt rất thấp (8-12) và CLAHE với clip_limit cao (3.0-4.0) là cần thiết.

     Nhiễu ảnh (noise) xuất phát từ nhiều nguồn. Sensor camera chất lượng thấp hoặc trong điều kiện thiếu sáng tạo ra nhiễu điểm ngẫu nhiên (salt-and-pepper noise) hoặc nhiễu Gaussian. Nén video với bitrate thấp tạo ra artifacts và làm giảm chất lượng, đặc biệt ở vùng chuyển động. Điều kiện môi trường như mưa, sương mù, bụi cũng gây nhiễu hình ảnh.

     Để giảm thiểu nhiễu, hệ thống sử dụng nhiều kỹ thuật. Gaussian blur làm mịn ảnh trước khi xử lý, giảm nhiễu cao tần nhưng có thể làm mờ biên. Median filter hiệu quả với nhiễu salt-and-pepper, giữ biên tốt hơn Gaussian. Morphological operations như opening loại bỏ các vùng nhiễu nhỏ, closing lấp các lỗ nhỏ trong đối tượng. Kết hợp các kỹ thuật này giúp cải thiện đáng kể chất lượng phát hiện.

     Chuyển động camera là một hạn chế quan trọng của hệ thống. Hệ thống được thiết kế với giả định camera cố định. Nếu camera di chuyển (rung, xoay, hoặc camera PTZ - Pan-Tilt-Zoom), toàn bộ khung hình thay đổi và thuật toán background subtraction sẽ coi toàn bộ là chuyển động. Điều này làm cho hệ thống không thể hoạt động đúng.

     Để xử lý camera di động, cần các kỹ thuật nâng cao như image stabilization (ổn định hình ảnh) để loại bỏ rung camera, camera motion compensation (bù chuyển động camera) ước tính và loại bỏ chuyển động do camera, hoặc sử dụng các thuật toán không dựa vào background model như object detection dựa trên deep learning. Tuy nhiên, các kỹ thuật này phức tạp hơn và yêu cầu tài nguyên xử lý lớn hơn.

     Bóng đổ (shadows) là nguồn gây false positives phổ biến. Khi đối tượng di chuyển, bóng của nó cũng di chuyển, và thuật toán motion detection có thể phát hiện cả bóng như một đối tượng riêng biệt. MOG2 có khả năng phát hiện và phân biệt bóng (giá trị 127 trong foreground mask) nhưng không hoàn hảo. Hệ thống cần lọc bỏ các pixel bóng trước khi xử lý tiếp hoặc sử dụng thông tin màu sắc HSV để phân biệt bóng và đối tượng thực.

[Bảng 1.2: Ảnh hưởng của các yếu tố môi trường đến hiệu năng hệ thống]
Yếu tố          | Điều kiện tốt     | Điều kiện trung bình | Điều kiện xấu
----------------|-------------------|----------------------|------------------
Độ phân giải    | 1080p+            | 720p                 | 480p hoặc thấp hơn
Ánh sáng        | Ban ngày ngoài trời| Trong nhà, tối      | Ban đêm không đèn
Nhiễu           | Thấp (camera tốt) | Trung bình           | Cao (thiếu sáng)
Chuyển động camera | Cố định hoàn toàn | Rung nhẹ          | PTZ hoặc camera di động
Detection rate  | 90-95 phần trăm   | 80-90 phần trăm      | 60-80 phần trăm
False positive  | 2-5 phần trăm     | 5-10 phần trăm       | 10-20 phần trăm
FPS đạt được    | 25-30             | 20-25                | 15-20


     Tóm lại, Chương 1 đã trình bày các cơ sở lý thuyết cần thiết cho hệ thống phát hiện xâm nhập, bao gồm tổng quan về xử lý ảnh số, các kỹ thuật phát hiện chuyển động (motion detection), ngưỡng hóa thích ứng (adaptive thresholding), phát hiện biên (edge detection), mở rộng vùng (region growing), phát hiện xâm nhập (intrusion detection), và các yếu tố ảnh hưởng đến chất lượng hệ thống. Những kiến thức này tạo nền tảng vững chắc cho việc triển khai hệ thống trong Chương 2.
