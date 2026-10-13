# Bộ câu hỏi hỏi đáp dự kiến: SLQ

Câu hỏi được chia theo chủ đề. Mỗi câu có một **câu trả lời gợi ý**: nói ngắn gọn trước, cô hỏi sâu thì giải thích thêm.

Mức độ: ⭐ cơ bản · ⭐⭐ trung bình · ⭐⭐⭐ khó hoặc dễ bị hỏi vặn.

---

## A. Hiểu bài báo

### 1. ⭐ Bài báo giải quyết vấn đề gì? Đóng góp chính là gì?
Bài báo giải quyết bài toán biến MLLM thành công cụ truy xuất ảnh–văn bản mà **không phải fine-tune**, tránh được chi phí lớn và tránh việc mô hình quên kiến thức cũ. Có ba đóng góp:
1. Phương pháp SLQ: học vài chục đến vài trăm nghìn tham số dưới dạng các vector query dùng chung.
2. Benchmark KARR-Bench để đánh giá truy xuất cần suy luận.
3. Kết quả tốt hơn LoRA và fine-tune toàn bộ trên COCO và Flickr30K, trong khi vẫn giữ được năng lực trả lời câu hỏi về ảnh (VQA).

### 2. ⭐ "Latent query" là gì? Khác gì với token thường?
Đó là một vector có thể học được, cùng số chiều với embedding của token (896 chiều với InternVL3-1B). Token thường được lấy từ bảng từ vựng. Latent query thì không ứng với từ nào; nó là tham số tự do, được tối ưu trực tiếp bằng gradient. Ta đưa nó vào mô hình như một token, và nó đóng vai trò "câu hỏi" để thu thập thông tin từ phần đầu vào.

### 3. ⭐⭐ "Shared" nghĩa là gì? Tại sao phải dùng chung?
**Cùng một bộ 20 query** được gắn vào cả chuỗi ảnh lẫn chuỗi văn bản. Dùng chung buộc hai phương thức đi qua cùng một "cổng" để tạo embedding, nên dễ rơi vào cùng một không gian. Ablation trong bài báo: dùng query riêng cho từng phương thức đạt 83,2 / 72,5, thấp hơn SLQ (84,9 / 74,3), dù có gấp đôi tham số.

### 4. ⭐⭐ Tại sao query phải đặt ở cuối chuỗi mà không đặt ở đầu?
MLLM là decoder-only, dùng **causal attention**: token thứ i chỉ nhìn được các token 1 đến i. Query đặt ở cuối thì nhìn được toàn bộ ảnh hoặc văn bản, nên tổng hợp được thông tin. Đặt ở đầu (như prompt tuning) thì query không thấy nội dung phía sau. Bài báo cho thấy prompt tuning chỉ đạt 82,2 / 69,7, so với 84,9 / 74,3 của SLQ.

### 5. ⭐⭐ Vì sao dùng mean pooling? Tại sao không lấy query cuối cùng?
Query cuối cùng nhìn được nhiều nhất, nhưng dồn hết vào một vector thì lại quay về nút thắt thông tin của cách "last token". Mean pooling cân bằng thông tin từ cả 20 vị trí. Ablation trong bài báo: Mean 84,9 / 74,3, Last 84,4 / 74,1, Max 84,1 / 73,8.

### 6. ⭐⭐ Giải thích hàm loss InfoNCE. Nhiệt độ τ có vai trò gì?
Trong một batch B cặp ảnh–caption, với mỗi ảnh, ta coi caption đúng là lớp đúng và B−1 caption còn lại là mẫu âm, rồi tính softmax và cross-entropy. Làm cả hai chiều I→T và T→I rồi lấy trung bình.

τ chia vào độ tương đồng trước softmax. τ nhỏ làm phân phối nhọn hơn, phạt mạnh các mẫu âm khó. Trong lần chạy của nhóm, τ tự học từ 0,07 xuống 0,021.

### 7. ⭐⭐ Batch size ảnh hưởng thế nào đến học tương phản?
Mẫu âm lấy từ chính các mẫu khác trong batch (in-batch negatives). Batch lớn thì có nhiều mẫu âm hơn, việc học khó hơn và hiệu quả hơn. Đây là lý do tác giả dùng batch 1024. Nhóm dùng 64 nên số mẫu âm ít hơn 16 lần; đây là một nguyên nhân khiến kết quả thấp hơn bài báo.

### 8. ⭐⭐ SLQ khác gì Q-Former của BLIP-2?
Q-Former là một **module transformer riêng**, có cross-attention và hàng trăm triệu tham số, đặt giữa bộ mã hóa ảnh và LLM. SLQ **không thêm module nào**: query đi thẳng vào MLLM và dùng chính attention có sẵn của mô hình. Số tham số chỉ là N × D.

### 9. ⭐⭐ SLQ khác gì LoRA?
LoRA thêm ma trận hạng thấp vào các lớp tuyến tính, nên **thay đổi cách mô hình tính toán** ở mọi lớp (8,7 đến 20 triệu tham số). SLQ không đụng vào trọng số nào; nó chỉ thay đổi **đầu vào**. Vì vậy SLQ giữ nguyên năng lực gốc. Bài báo còn cho thấy tăng rank LoRA từ 4 lên 32 thì kết quả **giảm** (84,5 → 82,1), nghĩa là càng sửa nhiều trọng số thì càng hại.

### 10. ⭐ KARR-Bench là gì? Vì sao cần nó?
Đây là benchmark gồm 2.915 cặp, viết lại caption COCO thành dạng câu đố cần tri thức hoặc suy luận, và cấm nêu tên vật thể. Ví dụ "con vật có (2+7) mạng" là con mèo. Benchmark này cần thiết vì COCO và Flickr chỉ kiểm tra khả năng khớp bề mặt, không đánh giá được năng lực suy luận vốn là thế mạnh của MLLM.

### 11. ⭐⭐⭐ Benchmark do GPT sinh ra thì có thiên vị (bias) không?
Có rủi ro đó. Tác giả giảm bớt bằng ba cách:
1. 4 người kiểm duyệt chéo, chỉ giữ lại 60–70% số mẫu.
2. Chia đều 6 loại suy luận.
3. Kiểm tra độ tương đồng với caption gốc, trung bình 0,46.

Tuy vậy, các mô hình có tri thức giống GPT vẫn có thể được lợi. Đây là một điểm có thể phản biện.

---

## B. Liên hệ môn học

### 12. ⭐ SLQ thuộc loại mô hình truy xuất nào trong môn học?
SLQ là **dense retrieval** theo kiểu **bi-encoder**: truy vấn và tài liệu được mã hóa **độc lập** thành vector, rồi xếp hạng theo cosine. Cách này tương tự mô hình không gian vector, nhưng vector là embedding học được chứ không phải TF-IDF.

### 13. ⭐⭐ Vì sao bi-encoder phù hợp với tìm kiếm quy mô lớn?
Vì embedding của toàn bộ kho ảnh được **tính trước một lần** rồi lập chỉ mục bằng công cụ tìm kiếm láng giềng gần đúng (ANN), ví dụ FAISS. Khi có truy vấn, chỉ cần mã hóa truy vấn rồi tìm vector gần nhất.

Cross-encoder đưa cặp truy vấn–tài liệu vào mô hình cùng lúc nên chính xác hơn, nhưng phải chạy mô hình cho **từng cặp**, không mở rộng được cho kho lớn.

### 14. ⭐ Recall@K là gì? Vì sao bài báo dùng Recall mà không dùng Precision?
Recall@K là tỷ lệ truy vấn có ít nhất một kết quả đúng nằm trong top-K. Mỗi truy vấn chỉ có một đáp án đúng (Text→Image), hoặc 5 caption đúng (Image→Text), nên điều quan trọng là đáp án có xuất hiện trong top-K hay không. Đây là thước đo chuẩn của các benchmark này, giúp so sánh trực tiếp với CLIP, BLIP và các mô hình khác.

### 15. ⭐⭐ Tại sao Image→Text thường cao hơn Text→Image?
Mỗi ảnh có **5 caption đúng**; chỉ cần 1 trong 5 lọt vào top-K là tính đúng. Còn mỗi caption chỉ có **1 ảnh đúng**. Ngoài ra, Image→Text tìm trong 5000 caption còn Text→Image tìm trong 1000 ảnh (Flickr), nên hai chiều không hoàn toàn đối xứng.

### 16. ⭐⭐ Có thể dùng thêm MRR hay nDCG không?
Có. Với chỉ một đáp án đúng, MRR (Mean Reciprocal Rank) phản ánh vị trí trung bình của đáp án. Nhóm đã lưu toàn bộ ma trận điểm nên tính thêm được dễ dàng. Nhóm dùng Recall@K để so sánh trực tiếp với bài báo.

---

## C. Phần tái hiện

### 17. ⭐ Nhóm đã làm gì cụ thể?
1. Đọc và phân tích bài báo.
2. Sửa code gốc vì không chạy được.
3. Huấn luyện 20 query trên 20 nghìn cặp COCO bằng GPU T4 của Kaggle, mất 155 phút.
4. Đánh giá zero-shot và sau huấn luyện trên Flickr30K-1K và COCO-5K.
5. So sánh với bài báo và với CLIP.

### 18. ⭐⭐ Code gốc có vấn đề gì?
- Import package `internvl3` nhưng repo không có.
- Dòng tạo `meta_queries` bị comment, nên chạy là báo lỗi.
- Mặc định 1 query và learning rate 1e-5, trong khi bài báo dùng 20 query và 5e-4.
- Không công bố checkpoint; các đường dẫn để trống.

Nhóm viết lại một script độc lập, giữ đúng tiền xử lý ảnh, prompt, vị trí query, mean pooling và loss.

### 19. ⭐⭐⭐ Làm sao chắc script của nhóm đúng với phương pháp gốc, không phải "chế" ra?
- So từng bước với `train_internvl_slq.py` và `eval_internvl_on_coco.py`: pad ảnh thành hình vuông màu xám 127, resize 448, 256 token ảnh, cùng template prompt, query gắn cuối, mean pooling, loss chia cho `logit_scale` khởi tạo 0,07.
- Chỗ khác duy nhất là phần kỹ thuật để chạy được trên T4: fp32, gradient checkpointing, batch nhỏ.
- Kết quả đạt 83–97% bài báo. Nếu sai phương pháp thì khó đạt mức này.

### 20. ⭐⭐⭐ Vì sao lần chạy đầu ra 0,1%? Làm sao phát hiện?
Kết quả 0,1% đúng bằng mức đoán ngẫu nhiên (1/1000), và Image→Text ra **giống hệt nhau** trước và sau huấn luyện. Điều đó cho thấy embedding bị hỏng chứ không phải do học kém. Khi chạy thử trên CPU với fp32, kết quả bình thường. Chuyển sang fp32 trên GPU thì hết lỗi.

Nhóm chưa tái hiện được NaN khi chạy fp16 trên CPU, nên nguyên nhân chính xác (có thể là tràn số trên nhân CUDA fp16 của T4) **chưa được chứng minh hoàn toàn**. Nhóm đã thêm bước kiểm tra NaN tự động vào code.

### 21. ⭐⭐ Vì sao không dùng bf16 như bài báo?
GPU T4 (kiến trúc Turing) không hỗ trợ bf16 bằng phần cứng; phải là GPU Ampere trở lên như A100 hay H100. Script tự chọn: bf16 nếu GPU hỗ trợ, ngược lại dùng fp32.

### 22. ⭐⭐ "FlashAttention2 is not installed" có ảnh hưởng không?
Không ảnh hưởng đến kết quả. FlashAttention chỉ là cách tính attention nhanh hơn, kết quả toán học vẫn như cũ. T4 không hỗ trợ FlashAttention2 nên nhóm dùng attention thường; chỉ chậm hơn và tốn bộ nhớ hơn.

### 23. ⭐⭐⭐ Bài báo ghi 36K tham số, nhóm ra 17.921. Ai sai?
Theo code tác giả công bố, tham số học được là `meta_queries` (20 × 896 = 17.920) cộng `logit_scale` (1), tức 17.921. Nhóm không tìm thấy thành phần nào khác trong code để ra 36K. Có thể tác giả tính theo cấu hình khác, hoặc số trong bài bị nhầm. Nhóm chỉ báo cáo con số đo được từ checkpoint.

### 24. ⭐⭐ Zero-shot lần 1 ra 3,8, lần 2 ra 7,2. Tại sao khác?
Zero-shot dùng 20 query **khởi tạo ngẫu nhiên**, và khi đánh giá không cố định seed, nên mỗi lần chạy có bộ query khác nhau. Điều này cũng cho thấy query ngẫu nhiên cho kết quả rất không ổn định, còn query đã học thì ổn định và cao hơn nhiều.

### 25. ⭐⭐ Zero-shot của nhóm có giống thí nghiệm thăm dò trong bài không?
Không hoàn toàn. Thí nghiệm thăm dò của bài dùng **1 query khởi tạo bằng 0** kèm prompt "Summary above sentence in one word". Zero-shot của nhóm dùng 20 query ngẫu nhiên và không có prompt tóm tắt. Nhóm dùng nó làm **mốc trước khi huấn luyện**, không phải để tái hiện thí nghiệm thăm dò.

### 26. ⭐⭐⭐ Đánh giá chỉ có một lần chạy, sao biết kết quả không do may mắn?
Đúng là nhóm chỉ huấn luyện một lần (seed 42) do giới hạn GPU. Tuy vậy:
1. Đánh giá trên tập lớn (5000 caption hoặc 25000 caption) nên sai số thống kê nhỏ.
2. Kết quả nhất quán trên cả hai bộ và mọi chỉ số.
3. Mức tăng rất lớn (7 → 69), khó là do ngẫu nhiên.

Nếu có thêm tài nguyên, nhóm sẽ chạy nhiều seed và báo cáo độ lệch chuẩn.

### 27. ⭐⭐ Huấn luyện trên COCO rồi đánh giá trên Flickr có hợp lệ không?
Hợp lệ. Bài báo cũng làm vậy. Đánh giá trên Flickr là kiểm tra khả năng **tổng quát hóa** sang bộ dữ liệu khác. Với COCO, nhóm huấn luyện trên COCO 2017 train và đánh giá trên COCO 5K test (Karpathy split); hai bộ ảnh này không trùng nhau.

### 28. ⭐⭐⭐ Ảnh của COCO 2017 train có trùng với COCO 5K test (Karpathy) không?
Đây là câu hỏi hay. COCO 2017 train được tạo từ 2014 train cộng phần lớn 2014 val. Karpathy test lại lấy 5000 ảnh từ 2014 val. Vì vậy **có khả năng có ảnh trùng** giữa tập train nhóm dùng và COCO 5K test. Nhóm **chưa kiểm tra** điều này.

Nếu có trùng, kết quả trên COCO có thể bị đẩy lên một chút. Hướng xử lý là lọc theo `cocoid` hoặc dùng Karpathy train split. Kết quả trên Flickr không bị ảnh hưởng.

### 29. ⭐⭐ Vì sao chỉ dùng 20 nghìn cặp mà không dùng hết?
Do giới hạn tài nguyên. Ở fp32, 20 nghìn cặp mất 155 phút trên T4. Toàn bộ khoảng 118 nghìn ảnh sẽ mất khoảng 15 giờ, vượt giới hạn 12 giờ mỗi lần chạy của Kaggle. Mỗi ảnh nhóm chỉ lấy 1 caption để trong batch không có hai mẫu dương trùng ảnh.

---

## D. Đánh giá phản biện

### 30. ⭐⭐ Điểm mạnh và yếu của SLQ?
**Mạnh:**
- Cực ít tham số, huấn luyện nhanh.
- Không làm hỏng mô hình gốc: một backbone có thể dùng chung cho nhiều tác vụ, mỗi tác vụ chỉ cần lưu một bộ query vài chục KB. File `slq_q20.pt` của nhóm chỉ 73 KB.
- Đơn giản, dễ cài đặt.

**Yếu:**
- Khi suy luận vẫn phải chạy cả MLLM (1 đến 8 tỷ tham số), nên chậm hơn nhiều so với CLIP.
- Không học được đặc trưng thị giác mới, nên khó dùng cho miền dữ liệu lạ.
- Mỗi ảnh tốn 256 token đầu vào.

### 31. ⭐⭐⭐ Nếu SLQ tốt như vậy, sao không dùng thay CLIP ngay?
Về **độ chính xác**, SLQ tốt hơn, nhất là với truy vấn cần suy luận. Về **chi phí khi suy luận**, mã hóa một ảnh với InternVL3-1B tốn hơn nhiều so với CLIP ViT-B (khoảng 150 triệu tham số). Với hệ thống cần độ trễ thấp, CLIP vẫn hợp lý hơn.

Kết quả của nhóm cũng cho thấy chiều Image→Text chưa vượt CLIP ViT-B. Lựa chọn phụ thuộc vào bài toán: cần độ chính xác và suy luận, hay cần tốc độ.

### 32. ⭐⭐ Tác giả tự nêu hạn chế gì?
1. Đóng băng backbone thì khó thích nghi với miền dữ liệu rất khác (ảnh y tế, ảnh vệ tinh). Có thể mở băng một phần hoặc thêm adapter cho phần thị giác.
2. Mới thử với InfoNCE đối xứng; chưa thử triplet loss, listwise hay hard negative mining.

### 33. ⭐⭐⭐ Có thể áp dụng cho tiếng Việt không?
Về nguyên lý là có, vì Qwen2 (LLM bên trong InternVL3) có hỗ trợ tiếng Việt ở mức nhất định. Nhóm đã viết sẵn chế độ đánh giá `--data custom` với file CSV gồm cột `image,caption`. Có thể thử trên UIT-ViIC hoặc KTVIC. Hiệu quả cần được kiểm chứng bằng thực nghiệm.

### 34. ⭐⭐ Nếu có thêm thời gian và GPU, nhóm sẽ làm gì?
1. Huấn luyện toàn bộ COCO, 5 epoch, batch lớn hơn (dùng GradCache hoặc nhiều GPU).
2. Làm lại thí nghiệm về số query N = 1, 5, 10, 20, 32.
3. Chạy nhiều seed.
4. So sánh với LoRA trong cùng điều kiện.
5. Thử trên dữ liệu tiếng Việt.

---

## E. Câu hỏi "bẫy" thường gặp

| Câu hỏi | Ý trả lời |
|---|---|
| "Mô hình có bị thay đổi gì không?" | Không. 100% trọng số InternVL3 giữ nguyên; chỉ học 20 vector query và τ. |
| "20 query là 20 mẫu dữ liệu à?" | Không. 20 query là 20 vector tham số. Dữ liệu huấn luyện là 20 nghìn cặp ảnh–caption. |
| "Kết quả có vượt bài báo không?" | Không, đạt khoảng 83–97%. Nguyên nhân là dữ liệu, batch và phần cứng ít hơn. |
| "Có vượt CLIP không?" | Chiều Text→Image vượt CLIP ViT-B trên cả hai bộ. Chiều Image→Text thấp hơn một chút. |
| "Nhóm có đánh giá KARR-Bench không?" | Không, vì bộ dữ liệu chưa được công bố. |
| "Code là của nhóm hay của tác giả?" | Logic theo đúng code tác giả. Nhóm viết lại để sửa lỗi và chạy được trên T4, có ghi rõ các điểm khác biệt. |
