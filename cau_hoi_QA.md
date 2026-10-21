# Bộ câu hỏi hỏi đáp dự kiến: SLQ và AdaSLQ

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

### 17. ⭐ Nhóm đã tái hiện như thế nào?
Nhóm chạy trên 4× A100 với đúng cấu hình bài báo: InternVL3-1B, 20 query, toàn bộ COCO 2017 train, 5 epoch, learning rate 5e-4, bf16, 1024 mẫu âm mỗi bước. Đánh giá trên Flickr30K-1K và COCO-5K test.

### 18. ⭐⭐ Code gốc có vấn đề gì?
- Import package `internvl3` nhưng repo không có.
- Dòng tạo `meta_queries` bị comment, nên chạy là báo lỗi.
- Mặc định 1 query và learning rate 1e-5, trong khi bài báo dùng 20 query và 5e-4.
- Không công bố checkpoint; các đường dẫn để trống.

Nhóm viết lại script nhưng giữ đúng tiền xử lý ảnh, prompt, vị trí query, mean pooling và loss, rồi mở rộng thành `slq_multilayer.py` để chạy nhiều GPU.

### 19. ⭐⭐ Kết quả tái hiện so với bài báo thế nào?
Nhóm đạt COCO I→T R@1 **60,10** so với 61,2 của bài báo, và COCO T→I R@1 **47,75** so với 48,5, tức khoảng 98%. Flickr I→T R@1 đạt 84,40, thấp hơn bài báo (86,7) 2,3 điểm. Riêng COCO I→T R@5 cao hơn một chút (85,12 so với 84,9).

### 20. ⭐⭐⭐ Làm sao chắc script của nhóm đúng với phương pháp gốc?
- So từng bước với `train_internvl_slq.py` và `eval_internvl_on_coco.py` của tác giả: pad ảnh thành hình vuông màu xám, resize 448, 256 token ảnh, cùng template prompt, query gắn cuối, mean pooling, loss chia cho `logit_scale` khởi tạo 0,07.
- Kết quả đạt khoảng 98% bài báo. Nếu sai phương pháp thì khó đạt mức này.

### 21. ⭐⭐ "FlashAttention2 is not installed" có ảnh hưởng không?
Không ảnh hưởng đến kết quả, vì FlashAttention chỉ là cách tính attention nhanh hơn. Nhóm dùng attention thường (eager), vì FlashAttention-2 gây sai khi chuỗi văn bản được đệm bên trái (left padding).

### 22. ⭐⭐⭐ Bài báo ghi khoảng 36K tham số, nhóm ra 17.921. Ai sai?
Theo code tác giả công bố, tham số học được là `meta_queries` (20 × 896 = 17.920) cộng `logit_scale` (1), tức 17.921. Các biến thể tiêm query ở tầng giữa của nhóm có thêm một LayerNorm (2 × 896 = 1.792), thành 19,7K. Nhóm không tìm thấy thành phần nào để ra 36K. Có thể tác giả tính theo cấu hình khác, hoặc con số trong bài bị nhầm.

### 23. ⭐⭐ Linear head có 803K tham số, sao lại kém SLQ nhiều vậy?
Linear head chỉ biến đổi **một vector** là token cuối. Vector đó vốn đã bị nút thắt thông tin, nên thêm bao nhiêu tham số phía sau cũng không lấy lại được thông tin đã mất. Query thì hoạt động **bên trong** MLLM, dùng attention để tự chọn thông tin cần lấy. Điều này khớp với ablation trong Bảng 5 của bài báo.

### 24. ⭐⭐ Zero-shot last token chỉ đạt khoảng 2%. Có phải code sai không?
Không. Token cuối của MLLM được huấn luyện để dự đoán từ tiếp theo, không phải để làm embedding truy xuất, nên khi chưa học gì thì vector của ảnh và văn bản chưa nằm cùng không gian. Bài báo cũng chỉ ra last token bị nút thắt thông tin. Con số này đóng vai trò mốc sàn.

### 25. ⭐⭐⭐ Ảnh COCO 2017 train có trùng với COCO 5K test (Karpathy) không?
Có khả năng. COCO 2017 train gồm 2014 train cộng phần lớn 2014 val, còn Karpathy test lấy 5000 ảnh từ 2014 val. Nhóm **chưa kiểm tra**. Nếu có trùng, kết quả COCO có thể cao hơn thực tế một chút. Cách khắc phục là lọc theo `cocoid` hoặc dùng Karpathy train split. Kết quả Flickr không bị ảnh hưởng.

Điều này ảnh hưởng như nhau tới baseline và AdaSLQ, nên so sánh tương đối giữa các cấu hình vẫn có ý nghĩa.

### 26. ⭐⭐ Batch "1024" của nhóm có đúng nghĩa như bài báo không?
Cần phân biệt. Số mẫu âm trong InfoNCE bằng batch trên mỗi GPU nhân số GPU, vì embedding được gom (all_gather) giữa các GPU. Tích lũy gradient (grad_accum) giúp batch hiệu dụng lớn hơn cho bước cập nhật, **nhưng không tăng số mẫu âm**. Cấu hình batch 256 trên mỗi GPU × 4 GPU thì có 1024 mẫu âm, tương đương bài báo.

---

## D. Phần cải tiến (AdaSLQ)

### 27. ⭐ AdaSLQ gồm những gì?
AdaSLQ gồm bốn thành phần, trên InternVL3-1B vẫn đóng băng:
1. **Prefix steering:** 5 query ở đầu chuỗi.
2. **Late Query Injection:** 15 query còn lại được tiêm vào tầng 12.
3. **Query thích ứng:** 15 query đó được sinh từ token cuối của tầng 11 qua một MLP.
4. **MR-4:** embedding = h₂₄ + α·MLP(h₂₁, h₂₂, h₂₃).

Kết quả: COCO I→T R@1 69,64, tức +9,5 so với SLQ tái lập.

### 28. ⭐⭐ Query đặt ở đầu không nhìn thấy đầu vào (causal), vậy nó có tác dụng gì?
Đúng là query ở đầu không nhìn thấy ảnh hay văn bản. Nhưng **mọi token phía sau nhìn thấy nó**, nên nó hoạt động như một "lời dặn" chung, định hướng cách mô hình xử lý đầu vào; giống prompt tuning. 15 query ở cuối vẫn làm nhiệm vụ tổng hợp. Thực nghiệm: 5 đầu + 15 cuối đạt 62,22, so với 60,10 khi đặt cả 20 ở cuối.

### 29. ⭐⭐ Tiêm query ở tầng 12 nghĩa là gì? Vì sao lại tốt hơn?
Ở tầng 1, ta đặt chỗ trống cho query. Khi chạy tới tầng 12, ta thay chỗ trống bằng vector query (qua một LayerNorm), rồi chạy tiếp tầng 12 đến 24. Query gặp ngay biểu diễn đã được 11 tầng xử lý, giàu ngữ nghĩa, thay vì phải tự đọc từ embedding thô. Cách này **không thêm tham số** ngoài LayerNorm, mà tăng khoảng 3,3 điểm.

Tiêm quá muộn (tầng 20) thì chỉ còn 4 tầng để tổng hợp, nên không còn lợi.

### 30. ⭐⭐ Tại sao lấy trung bình nhiều tầng không giúp, mà residual head lại giúp?
Các tầng có biểu diễn ở "thang đo" khác nhau; cộng trung bình đơn thuần làm lẫn thông tin. Residual head học cách lấy **phần bổ sung** từ tầng 21–23 rồi cộng vào tầng 24. Hệ số α khởi tạo bằng 0, nên lúc đầu mô hình giống hệt SLQ, sau đó mới học dần phần bổ sung. Kết quả: trung bình 4 tầng đạt 60,44, còn residual đạt 64,94.

### 31. ⭐⭐⭐ AdaSLQ có 10,59M tham số, nhiều hơn cả LoRA 1B (4,4M). Vậy còn đúng tinh thần SLQ không?
Đây là điểm yếu nhóm thừa nhận. AdaSLQ **vẫn giữ** ưu điểm quan trọng nhất là backbone đóng băng, không làm hỏng năng lực gốc, và một backbone dùng chung được cho nhiều tác vụ. Nhưng nó **không còn** lợi thế "vài chục nghìn tham số". Phần lớn tham số nằm ở MLP sinh query (lớp ra 512 → 15 × 896).

Nếu cần gọn nhẹ, LQI-12 là lựa chọn tốt nhất: 19,7K tham số mà vẫn tăng 3,3 điểm. Nhóm cũng chưa so sánh trực tiếp AdaSLQ với LoRA trong cùng điều kiện.

### 32. ⭐⭐⭐ Mỗi cấu hình chạy 1 lần, chọn trên tập test. Kết quả có đáng tin không?
Đây là hạn chế chính. Với các chênh lệch lớn (AdaSLQ +9,5, LQI +3,3, IAQ +2,9) thì khó là do ngẫu nhiên, vì tập test lớn (5000 ảnh, 25000 caption). Nhưng các chênh lệch nhỏ, như tầng 8, 12 và 16 chỉ cách nhau 0,1–0,3 điểm, **không đủ** để kết luận. Hướng sửa: tách tập validation từ COCO train để chọn cấu hình, và chạy 3 seed trở lên rồi báo cáo trung bình ± độ lệch chuẩn.

### 33. ⭐⭐ MoE query pool là gì? Sao không hiệu quả?
Ý tưởng là có một "ngân hàng" 30 hoặc 60 query; với mỗi đầu vào, một bộ định tuyến chọn 15 query phù hợp, cộng 5 query dùng chung. Kết quả chỉ đạt 63,28–63,74, gần như bằng LQI-12 tĩnh (63,40). Có thể do việc chọn rời rạc khó học hơn so với sinh liên tục bằng MLP. Nhóm chưa phân tích sâu hơn.

### 34. ⭐⭐ Các cải tiến có cần nhiều dữ liệu không?
Có khả năng. LQI và prefix gần như không thêm tham số nên ít phụ thuộc dữ liệu. MR-4 (3,23M) và bộ sinh query (khoảng 9,6M) thì nhiều tham số hơn hẳn, nên nếu dữ liệu ít sẽ dễ overfit. Nhóm chỉ thử trên toàn bộ COCO, chưa khảo sát theo lượng dữ liệu.

---

## E. Đánh giá phản biện

### 35. ⭐⭐ Điểm mạnh và yếu của SLQ?
**Mạnh:**
- Cực ít tham số, huấn luyện nhanh, không làm hỏng mô hình gốc.
- Mỗi tác vụ chỉ cần lưu một bộ query vài chục KB.

**Yếu:**
- Khi suy luận vẫn phải chạy cả MLLM, chậm hơn nhiều so với CLIP.
- Không học được đặc trưng thị giác mới, nên khó dùng cho miền dữ liệu lạ.

### 36. ⭐⭐⭐ Nếu SLQ tốt vậy, sao không dùng thay CLIP ngay?
Về độ chính xác, SLQ tốt hơn, nhất là với truy vấn cần suy luận. Về chi phí, mã hóa một ảnh bằng MLLM 1 tỷ tham số tốn hơn nhiều so với CLIP ViT-B (khoảng 150 triệu). Hệ thống cần độ trễ thấp vẫn nên dùng CLIP. Có thể kết hợp: CLIP lọc nhanh top-100, rồi SLQ xếp hạng lại.

### 37. ⭐⭐ Tác giả tự nêu hạn chế gì?
1. Đóng băng backbone thì khó thích nghi với miền rất khác (ảnh y tế, ảnh vệ tinh).
2. Mới thử với InfoNCE; chưa thử triplet loss, listwise hay hard negative mining.

### 38. ⭐⭐⭐ Có thể áp dụng cho tiếng Việt không?
Về nguyên lý là có, vì Qwen2 trong InternVL3 có hỗ trợ tiếng Việt ở mức nhất định. Script đã có chế độ `--data custom` với file CSV gồm cột `image,caption`. Có thể thử trên UIT-ViIC hoặc KTVIC, nhưng hiệu quả cần kiểm chứng.

---

## F. Câu hỏi "bẫy" thường gặp

| Câu hỏi | Ý trả lời |
|---|---|
| "Mô hình có bị thay đổi gì không?" | Không. 100% trọng số InternVL3 giữ nguyên, kể cả trong AdaSLQ. |
| "20 query là 20 mẫu dữ liệu à?" | Không. 20 query là 20 vector tham số. Dữ liệu huấn luyện là toàn bộ COCO. |
| "Tái hiện có đạt bài báo không?" | Đạt khoảng 98% ở COCO; Flickr I→T thấp hơn 2,3 điểm. |
| "AdaSLQ có vượt bài báo không?" | Có, khoảng +4 đến +8 điểm R@1, nhưng dùng 10,59M tham số, 1 seed, chọn trên tập test. |
| "Cải tiến nào đáng giá nhất?" | LQI-12: không thêm tham số mà tăng 3,3 điểm. |
| "Nhóm có đánh giá KARR-Bench không?" | Không, vì bộ dữ liệu chưa được công bố. |
| "Code là của nhóm hay của tác giả?" | Logic SLQ theo đúng code tác giả, nhóm sửa lỗi để chạy được. Các cải tiến AdaSLQ do nhóm viết (`slq_multilayer.py`). |
