# Kịch bản thuyết trình: SLQ (Shared Latent Queries)

- **Slide:** `bao_cao_SLQ.html`, 17 slide
- **Thời lượng gợi ý:** 15–18 phút trình bày và 5–10 phút hỏi đáp
- **Điều khiển:** phím ← → hoặc Space để chuyển slide, phím F để bật toàn màn hình
- **Phân công gợi ý:** 5 người, mỗi người một phần (xem bảng cuối file). Nhóm tự chỉnh lại cho phù hợp.

> Mỗi slide gồm: **Thời gian**, **Ý chính** (bắt buộc phải nói), **Lời thoại gợi ý** (nói tự nhiên, không đọc nguyên văn) và **Chuyển ý** sang slide sau.

---

## Slide 1: Trang bìa
**Thời gian:** 30 giây

**Ý chính:** chào cô, giới thiệu nhóm, nêu tên bài báo.

**Lời thoại gợi ý:**
> Em chào cô và các bạn. Nhóm em gồm 5 học viên: Kiên, Quang, Trang, Minh và Hưng. Hôm nay nhóm em trình bày bài báo **"SLQ: Bridging Modalities via Shared Latent Queries for Retrieval with Frozen MLLMs"**, được nhận đăng tại ICML 2026. Câu hỏi bài báo đặt ra là: có thể biến một mô hình ngôn ngữ lớn đa phương thức thành công cụ tìm kiếm ảnh–văn bản mà chỉ cần học khoảng 18 nghìn tham số hay không. Ngoài việc trình bày bài báo, nhóm em đã chạy lại code của tác giả và đánh giá kết quả.

**Chuyển ý:** "Trước hết, nhóm em xin nói về bài toán."

---

## Slide 2: Bài toán truy xuất đa phương thức
**Thời gian:** 1 phút

**Ý chính:**
- Định nghĩa: Text→Image (tìm ảnh từ câu) và Image→Text (tìm câu từ ảnh).
- Cách làm: dense retrieval, tức mã hóa thành vector rồi xếp hạng theo cosine similarity. Đây là nội dung liên quan trực tiếp đến môn học.
- Hai hướng hiện có: dual-encoder (CLIP) và MLLM làm retriever.

**Lời thoại gợi ý:**
> Bài toán là truy xuất đa phương thức: cho một câu mô tả thì tìm ảnh phù hợp, hoặc ngược lại. Cách phổ biến là dense retrieval, giống mô hình không gian vector đã học trong môn: mã hóa ảnh và văn bản thành vector trong cùng một không gian, rồi xếp hạng theo độ tương đồng cosine.
>
> Có hai hướng chính. Thứ nhất là dual-encoder như CLIP, dùng hai bộ mã hóa riêng cho ảnh và văn bản. Cách này nhanh nhưng chủ yếu khớp bề mặt. Thứ hai là dùng MLLM, các mô hình như GPT-4V hay Qwen-VL, vốn giàu tri thức và biết suy luận. Nhưng muốn dùng MLLM để tìm kiếm thì phải fine-tune.

**Chuyển ý:** "Vậy fine-tune thì có vấn đề gì?"

---

## Slide 3: Động lực
**Thời gian:** 1 phút

**Ý chính:** hai vấn đề của fine-tune, và giả thuyết của tác giả.

**Lời thoại gợi ý:**
> Có hai vấn đề. Một là **suy giảm ngữ nghĩa**: MLLM được huấn luyện để sinh văn bản, còn truy xuất cần phân biệt đúng sai. Hai mục tiêu này xung đột, nên cập nhật trọng số mạnh tay sẽ làm mô hình quên kiến thức cũ. Hai là **chi phí**: fine-tune toàn bộ bản 8B mất hơn 400 giờ GPU H800.
>
> Tác giả đưa ra giả thuyết: MLLM đã được huấn luyện trên lượng lớn cặp ảnh–văn bản, nên bên trong nó đã có sẵn một không gian ảnh–văn bản được căn chỉnh. Ta không cần dạy lại mô hình, chỉ cần "khơi gợi" biểu diễn đó ra.

**Chuyển ý:** "Để kiểm chứng giả thuyết này, tác giả làm một thí nghiệm thăm dò."

---

## Slide 4: Thí nghiệm thăm dò
**Thời gian:** 1 phút 15 giây

**Ý chính:**
- So sánh hai cách lấy embedding: last token và query token.
- Ba mức độ khó. Câu "(2+7) lives" là ví dụ đáng nhớ nhất.
- Kết luận: last token bị nút thắt thông tin.

**Lời thoại gợi ý:**
> Tác giả so sánh hai cách lấy vector từ MLLM khi chưa huấn luyện gì. Cách thứ nhất lấy token cuối cùng, cách mà VLM2VEC hay E5-V đang dùng. Cách thứ hai gắn thêm một vector query vào cuối chuỗi.
>
> Với câu đơn giản "A black cat", cả hai đều đúng. Với câu cần tri thức "An animal that barks", last token có điểm số các ảnh gần như nhau, nên khó phân biệt. Thú vị nhất là câu "con vật có (2+7) mạng". Mèo có 9 mạng, nên đáp án là con mèo. Last token tìm sai, còn query token tìm đúng.
>
> Kết luận: MLLM vốn đã biết suy luận. Vấn đề là last token phải nén mọi thứ vào một vector, gây ra nút thắt thông tin.

**Chuyển ý:** "Từ quan sát này, tác giả đề xuất SLQ."

---

## Slide 5: Kiến trúc SLQ (slide quan trọng nhất)
**Thời gian:** 2 phút

**Ý chính:**
- Gắn 20 vector query vào **cuối** chuỗi của cả ảnh lẫn văn bản.
- "Shared": dùng **cùng một** bộ query cho cả hai phương thức.
- MLLM đóng băng hoàn toàn. Lấy hidden state của 20 vị trí query, mean pooling, rồi chuẩn hóa L2.
- Loss: InfoNCE đối xứng.

**Lời thoại gợi ý:**
> Đây là phần cốt lõi. Với ảnh, chuỗi đầu vào gồm 256 token ảnh, câu prompt, rồi 20 vector query. Với văn bản cũng vậy: văn bản, prompt, rồi **đúng 20 vector query đó**. Chữ "Shared" nghĩa là hai phương thức dùng chung một bộ query, nhờ vậy cả hai bị ép về cùng một không gian.
>
> Toàn bộ MLLM bị đóng băng, không đổi một trọng số nào. Chạy qua mô hình xong, ta lấy hidden state lớp cuối tại 20 vị trí query, lấy trung bình, rồi chuẩn hóa. Kết quả là vector embedding.
>
> Hàm mất mát là InfoNCE đối xứng. Trong một batch, kéo cặp ảnh–caption đúng lại gần nhau và đẩy các cặp sai ra xa. Những thứ duy nhất được học là 20 vector query và nhiệt độ τ.

**Chuyển ý:** "Một câu hỏi tự nhiên là: tại sao query phải đặt ở cuối?"

---

## Slide 6: Vì sao đặt query ở cuối?
**Thời gian:** 1 phút

**Ý chính:** causal attention, minh họa bằng hình lưới attention trên slide.

**Lời thoại gợi ý:**
> MLLM là mô hình decoder-only, dùng causal attention: mỗi token chỉ nhìn được các token đứng trước nó. Hình bên trái là cách prompt tuning làm: đặt token học được ở đầu chuỗi. Khi đó các ô màu hồng chỉ nhìn được chính nó, không thấy nội dung phía sau. Hình bên phải là SLQ: query đặt ở cuối nên nhìn được toàn bộ đầu vào. Vì vậy query thu thập được toàn bộ thông tin.
>
> Thí nghiệm ablation trong bài báo xác nhận điều này: prompt tuning kém hơn SLQ khoảng 3–4 điểm.

**Chuyển ý:** "Ngoài phương pháp, bài báo còn đóng góp một benchmark mới."

---

## Slide 7: KARR-Bench
**Thời gian:** 1 phút

**Ý chính:**
- Benchmark cũ chỉ kiểm tra khớp bề mặt; KARR-Bench kiểm tra suy luận.
- Gồm 2.915 cặp, có 6 loại suy luận, tên vật thể bị che đi.

**Lời thoại gợi ý:**
> Các benchmark như COCO dùng caption mô tả trực tiếp, kiểu "con mèo xám ngồi trên kệ sách". KARR-Bench viết lại thành câu đố: "con vật có (2+7) mạng ngồi trên kệ sách". Câu đố do GPT-5-mini sinh ra, sau đó 4 người kiểm duyệt chéo. Quy tắc là cấm nêu tên vật thể. Bộ dữ liệu có 2.915 cặp, chia khá đều cho 6 loại suy luận: văn hóa, công dụng, không gian, chức năng, bách khoa và logic toán.
>
> Bộ này **chưa được công bố**, nên nhóm em không đánh giá được trên nó.

**Chuyển ý:** "Giờ ta xem kết quả trong bài báo."

---

## Slide 8: Kết quả trong bài báo
**Thời gian:** 1 phút 15 giây

**Ý chính:**
- SLQ-8B dùng 144K tham số, so với 20,2 triệu của LoRA, tức ít hơn khoảng 140 lần. Giờ GPU giảm 70%.
- Kết quả vẫn tốt hơn, và **giữ nguyên năng lực VQA**.

**Lời thoại gợi ý:**
> Với bản 8B, SLQ chỉ học 144 nghìn tham số, còn LoRA cần 20 triệu và fine-tune toàn bộ cần 7,6 tỷ. Thời gian huấn luyện giảm 70% so với LoRA, nhưng kết quả truy xuất vẫn cao hơn.
>
> Điểm đáng chú ý nhất là ba cột MMMU, RealWorldQA và OCRBench, đo khả năng hiểu ảnh của mô hình gốc. Fine-tune và LoRA đều làm các điểm này giảm, tức mô hình bị "quên". SLQ thì giữ nguyên, vì backbone không bị động vào. Trên benchmark MMEB gồm 36 tác vụ, SLQ-8B đạt 67,5 điểm, cao nhất trong bảng so sánh.

**Chuyển ý:** "Tại sao lại là 20 query?"

---

## Slide 9: Ablation, chọn N = 20
**Thời gian:** 1 phút

**Ý chính:** N tăng đến 20 thì tốt lên, N = 32 thì giảm. Query dùng chung tốt hơn query riêng. Mean pooling tốt nhất.

**Lời thoại gợi ý:**
> Tác giả thử số query từ 1 đến 32. Kết quả tăng dần và đạt đỉnh ở N = 20, sang 32 thì giảm vì dư thừa và overfit. Có hai phát hiện khác. Một là dùng chung query tốt hơn mỗi phương thức một bộ riêng, dù bộ riêng có gấp đôi tham số. Hai là mean pooling tốt hơn max pooling hay lấy query cuối.

**Chuyển ý:** "Đó là phần bài báo. Tiếp theo là phần nhóm em tự làm: tái hiện kết quả."

---

## Slide 10: Thiết lập tái hiện
**Thời gian:** 1 phút

**Ý chính:**
- Giữ nguyên backbone, số query và learning rate.
- Thu nhỏ dữ liệu, batch và phần cứng.
- Mô hình chỉ học **17.921 tham số**. τ học được là 0,021.

**Lời thoại gợi ý:**
> Nhóm em giữ đúng backbone InternVL3-1B, 20 query và learning rate 5e-4. Do chỉ có GPU T4 miễn phí của Kaggle, nhóm em thu nhỏ quy mô: 20 nghìn cặp ảnh–caption của COCO thay vì toàn bộ, 1 epoch thay vì 5, batch 64 thay vì 1024.
>
> Số tham số huấn luyện là 17.921, đúng bằng 20 nhân 896 cộng 1 cho τ. Bài báo ghi khoảng 36 nghìn cho bản 1B. Nhóm em đã đọc kỹ code tác giả công bố và chỉ tính ra được khoảng 18 nghìn.

**Chuyển ý:** "Quá trình huấn luyện diễn ra như sau."

---

## Slide 11: Quá trình huấn luyện
**Thời gian:** 1 phút

**Ý chính:**
- Loss giảm từ 3,63 xuống khoảng 0,2–0,3 sau khoảng 100 bước.
- τ giảm từ 0,07 xuống 0,021.
- Huấn luyện mất 155 phút trên T4.

**Lời thoại gợi ý:**
> Đường màu hồng là loss. Nó bắt đầu ở 3,63, gần mức đoán ngẫu nhiên ln 64 ≈ 4,16, rồi giảm rất nhanh và ổn định quanh 0,2–0,3 sau khoảng 100 bước. Đường xanh là nhiệt độ τ, mô hình tự học từ 0,07 xuống 0,021. τ nhỏ hơn làm phân phối "nhọn" hơn, tức mô hình tự tin hơn khi phân biệt đúng sai. Huấn luyện mất 155 phút; cả quá trình gồm đánh giá mất khoảng 3,6 giờ.

**Chuyển ý:** "Tuy nhiên để có được kết quả này, nhóm em đã gặp khá nhiều trục trặc."

---

## Slide 12: Những trục trặc khi tái hiện
**Thời gian:** 1 phút 30 giây

**Ý chính:**
- Code gốc không chạy được: thiếu package, dòng tạo query bị comment, cấu hình mặc định khác bài báo.
- Lỗi fp16: kết quả 0,1%, ngang đoán ngẫu nhiên. Chuyển sang fp32 thì kết quả bình thường.

**Lời thoại gợi ý:**
> Code tác giả công bố không chạy được ngay. Code import một package không có trong repo. Dòng tạo các vector query bị comment, nên chạy là báo lỗi. Cấu hình mặc định cũng khác bài báo: 1 query thay vì 20, learning rate 1e-5 thay vì 5e-4. Nhóm em viết lại thành một script độc lập nhưng giữ đúng logic gốc.
>
> Lỗi khó nhất: lần chạy đầu ra kết quả **0,1%**, ngang đoán bừa, kể cả sau khi huấn luyện. Nhóm em phát hiện khi chạy trên GPU T4 với số thực 16 bit (fp16), kết quả bị hỏng. Chuyển sang fp32 thì mọi thứ bình thường. Bài học là luôn so với mức đoán ngẫu nhiên để phát hiện lỗi, và nhóm em đã thêm bước tự kiểm tra NaN vào code.

**Chuyển ý:** "Và đây là kết quả."

---

## Slide 13: Trước và sau khi học
**Thời gian:** 1 phút

**Ý chính:** chỉ học 17.921 tham số mà Text→Image R@1 tăng từ 7,2 lên 69,0, gần 10 lần.

**Lời thoại gợi ý:**
> Thanh xám là kết quả zero-shot: 20 query còn ngẫu nhiên, chưa học. Thanh màu là sau khi học. Text→Image R@1, tức tỷ lệ tìm đúng ảnh ngay ở vị trí đầu tiên, tăng từ 7,2% lên 69%. Image→Text R@1 tăng từ 17,7% lên 75,7%. R@10 đạt 93,7% và 98,5%. Toàn bộ thay đổi này chỉ nhờ 17.921 tham số; mô hình 1 tỷ tham số không đổi gì.

**Chuyển ý:** "So với bài báo và CLIP thì sao?"

---

## Slide 14: So sánh
**Thời gian:** 1 phút 15 giây

**Ý chính:**
- Kết quả đạt khoảng 83–97% số liệu bài báo.
- Chiều Text→Image vượt CLIP ViT-B trên cả hai bộ. Riêng COCO cao hơn 11,6 điểm.
- Chiều Image→Text gần bằng CLIP ViT-B (75,7 so với 77,8; 50,9 so với 51,0).

**Lời thoại gợi ý:**
> Dòng tô sáng là của nhóm em. So với bài báo, nhóm em đạt khoảng 83 đến 97 phần trăm tùy chỉ số. Ví dụ Flickr Text→Image R@5 đạt 89,9, so với 92,9 của bài báo.
>
> So với CLIP ViT-B, ở chiều Text→Image nhóm em vượt trên cả hai bộ; riêng COCO R@1 cao hơn 11,6 điểm. Ở chiều Image→Text thì gần bằng, thấp hơn một chút. Nhóm em nói rõ điều này để không phóng đại kết quả.

**Chuyển ý:** "Vì sao vẫn thấp hơn bài báo?"

---

## Slide 15: Phân tích chênh lệch
**Thời gian:** 1 phút

**Ý chính:** ít dữ liệu, batch nhỏ, phần cứng yếu. Xu hướng chính vẫn được tái hiện.

**Lời thoại gợi ý:**
> Có ba nguyên nhân chính. Một: nhóm em chỉ dùng khoảng 20 nghìn cặp và 1 epoch, so với toàn bộ COCO và 5 epoch của bài báo. Hai: batch 64 so với 1024. Với InfoNCE, mỗi mẫu so với các mẫu khác trong batch làm mẫu âm, nên batch nhỏ hơn 16 lần thì tín hiệu học yếu hơn. Ba: một GPU T4 chạy fp32 và quota Kaggle giới hạn.
>
> Dù vậy, kết luận chính của bài báo đã được tái hiện: chỉ học vài chục nghìn tham số trên một mô hình đóng băng vẫn cho kết quả truy xuất rất tốt.

**Chuyển ý:** "Nhóm em xin tổng kết."

---

## Slide 16: Kết luận
**Thời gian:** 1 phút

**Ý chính:** đã làm được gì, hạn chế, hướng tiếp theo.

**Lời thoại gợi ý:**
> Nhóm em đã phân tích phương pháp SLQ và KARR-Bench, sửa code gốc và tái hiện thành công trên GPU miễn phí, đạt tới 96,7% kết quả bài báo ở Flickr Text→Image R@5.
>
> Hạn chế: chưa đánh giá được trên KARR-Bench vì bộ dữ liệu chưa công bố, dữ liệu huấn luyện ít, và chỉ thử bản 1B. Bản thân phương pháp cũng có hạn chế: đóng băng backbone thì khó thích nghi với ảnh y tế hay ảnh vệ tinh.
>
> Hướng tiếp theo: huấn luyện trên toàn bộ COCO, làm lại thí nghiệm về số query, và thử trên dữ liệu tiếng Việt như UIT-ViIC.

---

## Slide 17: Cảm ơn và hỏi đáp
**Thời gian:** 10 giây

> Nhóm em xin cảm ơn cô và các bạn đã lắng nghe. Nhóm em rất mong nhận được câu hỏi và góp ý ạ.

---

## Phân công gợi ý

| Phần | Slide | Người trình bày | Thời gian |
|---|---|---|---|
| Mở đầu và bài toán | 1–3 | Hoàng Văn Kiên | ~2,5 phút |
| Thăm dò và phương pháp | 4–6 | Nguyễn Minh Quang | ~4,5 phút |
| KARR-Bench và kết quả bài báo | 7–9 | Nguyễn Thị Quỳnh Trang | ~3,5 phút |
| Tái hiện: thiết lập, huấn luyện, lỗi | 10–12 | Đỗ Nhật Minh | ~3,5 phút |
| Kết quả, phân tích, kết luận | 13–17 | Trần Khánh Hưng | ~4,5 phút |

## Mẹo khi trình bày
- **Slide 5 và slide 12** là hai chỗ cô dễ hỏi nhất. Người trình bày nên nắm thật chắc.
- Nhấn mạnh con số **17.921 tham số** và câu đố **"(2+7) lives"**. Hai chi tiết này dễ nhớ.
- Khi nói về kết quả, **nói thẳng** những chỗ thấp hơn bài báo hoặc thấp hơn CLIP (chiều Image→Text). Thẳng thắn về hạn chế được đánh giá cao hơn là phóng đại.
- Mở sẵn file `bao_cao_SLQ.html` trên máy làm phương án dự phòng, để không phụ thuộc mạng.
