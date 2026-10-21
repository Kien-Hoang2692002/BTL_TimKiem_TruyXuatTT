# Kịch bản thuyết trình: SLQ và AdaSLQ

- **Slide:** `bao_cao_SLQ.html`, 18 slide, gồm 3 phần: bài báo (1–9), tái hiện (10–11), cải tiến của nhóm (12–17).
- **Thời lượng gợi ý:** 18–20 phút trình bày và 5–10 phút hỏi đáp
- **Điều khiển:** phím ← → hoặc Space để chuyển slide, phím F để bật toàn màn hình
- **Phân công gợi ý:** 5 người (xem bảng cuối file). Nhóm tự chỉnh lại cho phù hợp.

> Mỗi slide gồm: **Thời gian**, **Ý chính** (bắt buộc phải nói), **Lời thoại gợi ý** (nói tự nhiên, không đọc nguyên văn) và **Chuyển ý** sang slide sau.

---

## Slide 1: Trang bìa
**Thời gian:** 30 giây

**Ý chính:** chào cô, giới thiệu nhóm, nêu tên bài báo và ba phần của bài trình bày.

**Lời thoại gợi ý:**
> Em chào cô và các bạn. Nhóm em gồm 5 học viên: Kiên, Quang, Trang, Minh và Hưng. Nhóm em trình bày bài báo **"SLQ: Bridging Modalities via Shared Latent Queries for Retrieval with Frozen MLLMs"**, ICML 2026. Bài trình bày có ba phần: nội dung bài báo, phần nhóm em tái hiện lại kết quả, và một số cải tiến nhóm em đề xuất, gọi là AdaSLQ.

**Chuyển ý:** "Trước hết là bài toán."

---

## Slide 2: Bài toán truy xuất đa phương thức
**Thời gian:** 1 phút

**Ý chính:**
- Text→Image và Image→Text.
- Dense retrieval: mã hóa thành vector, xếp hạng theo cosine.
- Hai hướng: dual-encoder (CLIP) và MLLM làm retriever.

**Lời thoại gợi ý:**
> Bài toán là cho một câu thì tìm ảnh phù hợp, hoặc ngược lại. Cách phổ biến là dense retrieval, giống mô hình không gian vector trong môn học: mã hóa ảnh và văn bản thành vector trong cùng một không gian, rồi xếp hạng theo cosine.
>
> Có hai hướng. Dual-encoder như CLIP thì nhanh nhưng chủ yếu khớp bề mặt. MLLM thì giàu tri thức và biết suy luận, nhưng muốn dùng để tìm kiếm thì phải fine-tune.

**Chuyển ý:** "Vậy fine-tune có vấn đề gì?"

---

## Slide 3: Động lực
**Thời gian:** 1 phút

**Ý chính:** suy giảm ngữ nghĩa, chi phí lớn, và giả thuyết "khơi gợi" của tác giả.

**Lời thoại gợi ý:**
> Thứ nhất, MLLM được huấn luyện để sinh văn bản, còn truy xuất cần phân biệt đúng sai. Cập nhật trọng số mạnh tay sẽ làm mô hình quên kiến thức cũ. Thứ hai là chi phí: fine-tune toàn bộ bản 8B mất hơn 400 giờ GPU H800.
>
> Giả thuyết của tác giả: MLLM đã có sẵn không gian ảnh–văn bản được căn chỉnh. Ta chỉ cần "khơi gợi" nó ra, không cần dạy lại.

**Chuyển ý:** "Tác giả kiểm chứng bằng một thí nghiệm thăm dò."

---

## Slide 4: Thí nghiệm thăm dò
**Thời gian:** 1 phút

**Ý chính:** last token và query token. Ví dụ "(2+7) lives". Nút thắt thông tin.

**Lời thoại gợi ý:**
> Tác giả so sánh hai cách lấy vector khi chưa huấn luyện: lấy token cuối, và gắn thêm một query vào cuối chuỗi. Câu đơn giản thì cả hai đều đúng. Câu đố "con vật có (2+7) mạng" thì last token tìm sai, còn query tìm đúng con mèo. Kết luận: MLLM vốn biết suy luận, nhưng last token phải nén mọi thứ vào một vector nên bị nút thắt thông tin.

**Chuyển ý:** "Từ đó tác giả đề xuất SLQ."

---

## Slide 5: Kiến trúc SLQ
**Thời gian:** 1 phút 30 giây

**Ý chính:**
- 20 query gắn vào **cuối** chuỗi, **dùng chung** cho ảnh và văn bản.
- MLLM đóng băng. Lấy hidden state của 20 vị trí query, mean pooling, chuẩn hóa L2.
- Loss InfoNCE đối xứng. Chỉ học query và nhiệt độ τ.

**Lời thoại gợi ý:**
> Với ảnh, chuỗi gồm 256 token ảnh, prompt, rồi 20 query. Với văn bản cũng vậy, và dùng **đúng 20 query đó**, nên mới gọi là "Shared". MLLM bị đóng băng hoàn toàn. Ta lấy hidden state lớp cuối tại 20 vị trí query, lấy trung bình, chuẩn hóa, được vector embedding. Loss là InfoNCE: kéo cặp đúng lại gần, đẩy cặp sai ra xa.

**Chuyển ý:** "Tại sao query phải ở cuối?"

---

## Slide 6: Vì sao đặt query ở cuối?
**Thời gian:** 45 giây

**Lời thoại gợi ý:**
> MLLM dùng causal attention: mỗi token chỉ nhìn được các token đứng trước. Query ở cuối thì nhìn thấy toàn bộ đầu vào, như hình bên phải. Prompt tuning đặt ở đầu nên không thấy gì phía sau, và kém hơn SLQ khoảng 3–4 điểm.
>
> Ghi nhớ ý này, vì ở phần cải tiến nhóm em sẽ thử đặt một phần query ở đầu với một vai trò khác.

**Chuyển ý:** "Bài báo còn có một benchmark mới."

---

## Slide 7: KARR-Bench
**Thời gian:** 45 giây

**Lời thoại gợi ý:**
> KARR-Bench viết lại caption COCO thành câu đố cần tri thức, cấm nêu tên vật thể. Bộ này có 2.915 cặp và 6 loại suy luận. Bộ dữ liệu **chưa được công bố**, nên nhóm em không đánh giá được trên nó.

---

## Slide 8: Kết quả trong bài báo
**Thời gian:** 1 phút

**Ý chính:** SLQ-8B dùng 144K tham số, so với 20,2M của LoRA. Giờ GPU giảm 70%. Kết quả tốt hơn và giữ nguyên năng lực VQA.

**Lời thoại gợi ý:**
> Bản 8B chỉ học 144 nghìn tham số, ít hơn LoRA khoảng 140 lần và nhanh hơn 70%, nhưng kết quả vẫn cao hơn. Điểm quan trọng: các chỉ số hiểu ảnh như MMMU và OCRBench được giữ nguyên, còn LoRA và fine-tune làm chúng giảm.

---

## Slide 9: Ablation, chọn N = 20
**Thời gian:** 45 giây

**Lời thoại gợi ý:**
> Kết quả tốt nhất ở 20 query; 32 query thì giảm vì dư thừa. Query dùng chung tốt hơn query riêng cho từng phương thức. Mean pooling tốt nhất.

**Chuyển ý:** "Đó là bài báo. Phần tiếp theo là nhóm em tái hiện lại."

---

## Slide 10: Thiết lập tái hiện
**Thời gian:** 1 phút 15 giây

**Ý chính:**
- Chạy đúng cấu hình bài báo: toàn bộ COCO, 5 epoch, bf16, 4× A100, 1024 mẫu âm.
- Mô hình học 17.921 tham số (19,7K khi có thêm LayerNorm). Bài báo ghi ~36K.
- Code gốc không chạy được ngay; nhóm sửa và mở rộng thành `slq_multilayer.py`.

**Lời thoại gợi ý:**
> Nhóm em tái hiện trên 4 GPU A100, giữ đúng cấu hình bài báo: InternVL3-1B, 20 query, toàn bộ COCO, 5 epoch, learning rate 5e-4, bf16. Mỗi GPU xử lý 256 mẫu và embedding được gom giữa 4 GPU, nên mỗi bước có 1024 mẫu âm như bài báo.
>
> Số tham số đo được là 17.921, tức 20 nhân 896 cộng 1. Bài báo ghi khoảng 36 nghìn, nhưng code công bố chỉ ra khoảng 18 nghìn.
>
> Code tác giả không chạy được ngay: thiếu package, dòng tạo query bị comment, cấu hình mặc định khác bài báo. Nhóm em viết lại giữ đúng logic gốc, sau đó mở rộng để chạy nhiều GPU và thử các cải tiến.

**Chuyển ý:** "Kết quả tái hiện như sau."

---

## Slide 11: Kết quả tái hiện
**Thời gian:** 1 phút 15 giây

**Ý chính:**
- Tái hiện đạt khoảng 97–99% bài báo, thấp hơn 0,7–2,3 điểm R@1.
- Linear head 803K tham số kém xa SLQ, khớp với Bảng 5 của bài báo.

**Lời thoại gợi ý:**
> Hai dòng đầu là mốc đối chứng. Zero-shot lấy token cuối gần như ngẫu nhiên, khoảng 2%. Thêm một linear head 803 nghìn tham số lên token cuối cũng chỉ đạt 37%. Trong khi đó SLQ với khoảng 20 nghìn tham số đạt 60%. Điều này xác nhận vai trò của query.
>
> Bản tái lập của nhóm em đạt 60,1 ở COCO I→T R@1, so với 61,2 của bài báo, tức khoảng 98%. Flickr I→T thấp hơn bài báo 2,3 điểm. Nhóm em coi như đã tái hiện thành công.

**Chuyển ý:** "Từ nền này, nhóm em thử cải tiến SLQ."

---

## Slide 12: Bốn hướng cải tiến
**Thời gian:** 1 phút 30 giây

**Ý chính:** prefix steering, tiêm query ở tầng giữa (LQI), lấy nhiều tầng kèm residual (MR-4), query thích ứng (IAQ). Gộp lại thành AdaSLQ. Backbone vẫn đóng băng.

**Lời thoại gợi ý:**
> Nhóm em thử bốn hướng.
> - **Một, prefix steering:** đặt 5 query ở **đầu** chuỗi. Chúng không nhìn thấy đầu vào, nhưng mọi token phía sau nhìn thấy chúng, nên chúng giống một "lời dặn" chung về cách đọc đầu vào.
> - **Hai, Late Query Injection:** thay vì đưa query vào từ tầng 1, ta tiêm vào tầng 12, nơi biểu diễn đã giàu ngữ nghĩa.
> - **Ba, MR-4:** không chỉ lấy tầng 24 mà lấy thêm tầng 21 đến 23, qua một MLP residual. Hệ số α khởi tạo bằng 0, nên lúc đầu mô hình giống hệt SLQ.
> - **Bốn, query thích ứng:** query không cố định mà được sinh từ token cuối của tầng 11, nên mỗi ảnh hoặc câu có bộ query riêng.
>
> AdaSLQ là tổ hợp cả bốn, như sơ đồ bên trái. InternVL3 vẫn đóng băng 100%.

---

## Slide 13: Ablation vị trí query
**Thời gian:** 1 phút

**Ý chính:**
- Tiêm ở tầng 8–16 tăng khoảng 3,3 điểm và **không tốn thêm tham số**. Tầng 20 thì không giúp gì.
- 5 query đầu + 15 query cuối là tốt nhất trong các tỉ lệ thử.
- Lưu ý: trục biểu đồ bắt đầu từ 58.

**Lời thoại gợi ý:**
> Biểu đồ trái: tiêm ở tầng 8, 12 hay 16 đều tăng khoảng 3 điểm so với tầng 1, và không thêm tham số nào. Tiêm muộn ở tầng 20 thì không còn đủ tầng để tổng hợp thông tin. Ba tầng 8, 12, 16 chênh nhau rất ít; nhóm em chọn tầng 12 vì chiều T→I tốt nhất.
>
> Biểu đồ phải: 5 query đầu cộng 15 query cuối tốt nhất, tăng khoảng 2 điểm. Cô lưu ý giúp em trục bắt đầu từ 58 nên chênh lệch trông lớn hơn thực tế.

---

## Slide 14: Ablation trích xuất và thích ứng
**Thời gian:** 1 phút

**Ý chính:**
- Lấy trung bình nhiều tầng không giúp. Cần head có học (residual) mới tăng, lên 64,94.
- Query thích ứng bằng MLP tăng khoảng 3 điểm. MoE thì gần như không giúp.

**Lời thoại gợi ý:**
> Bảng trái: nếu chỉ lấy trung bình 2 hay 4 tầng cuối thì kết quả gần như không đổi. Phải có một residual head học được thì mới tăng, lên 64,94. Đổi lại, head này tốn 3,2 triệu tham số.
>
> Bảng phải: sinh query theo từng đầu vào bằng MLP tăng từ 63,4 lên 66,3. Nhóm em cũng thử chọn query từ một "ngân hàng" kiểu Mixture-of-Experts, nhưng gần như không cải thiện.

---

## Slide 15: AdaSLQ
**Thời gian:** 1 phút

**Ý chính:**
- AdaSLQ đạt COCO I→T R@1 69,64, tức +9,5 so với SLQ tái lập. Flickr I→T đạt 91,0.
- Mỗi dòng "Chỉ…" là thí nghiệm riêng, không cộng dồn.
- Đổi lại: 10,59M tham số, gấp 537 lần.

**Lời thoại gợi ý:**
> Gộp các hướng lại, AdaSLQ đạt 69,64 ở COCO I→T R@1, cao hơn SLQ tái lập 9,5 điểm. Các chỉ số khác tăng 6–7 điểm. Mỗi dòng "Chỉ…" ở giữa là một thí nghiệm riêng so với baseline, không phải cộng dồn từng bước.
>
> Cái giá phải trả là số tham số: 10,59 triệu, gấp khoảng 537 lần SLQ.

**Chuyển ý:** "Nhóm em muốn nói rõ một số điểm khi đánh giá kết quả này."

---

## Slide 16: Đánh giá phản biện
**Thời gian:** 1 phút 15 giây

**Ý chính:**
- AdaSLQ nhiều tham số hơn cả LoRA 1B. LQI-12 là cải tiến "miễn phí".
- 1 seed, chọn cấu hình trên tập test.
- Có thể trùng ảnh giữa COCO train và test.
- Chưa so với LoRA trong cùng điều kiện.

**Lời thoại gợi ý:**
> Thứ nhất, AdaSLQ có 10,6 triệu tham số, nhiều hơn cả LoRA bản 1B. Nó vẫn không làm hỏng backbone, nhưng không còn lợi thế "vài chục nghìn tham số". Nếu cần gọn nhẹ thì LQI-12 là lựa chọn tốt: không thêm tham số mà vẫn tăng 3,3 điểm.
>
> Thứ hai, mỗi cấu hình mới chạy một lần, và nhóm em chọn cấu hình tốt nhất dựa trên chính tập test. Vì vậy những chênh lệch dưới 0,5 điểm chưa đủ để kết luận.
>
> Thứ ba, COCO 2017 train có thể trùng ảnh với COCO 5K test, nhóm em chưa lọc. Cuối cùng, nhóm em chưa chạy LoRA hay fine-tune trong cùng điều kiện để so trực tiếp với AdaSLQ.

---

## Slide 17: Kết luận
**Thời gian:** 45 giây

**Lời thoại gợi ý:**
> Tóm lại, nhóm em đã phân tích bài báo, sửa code và tái hiện SLQ trên toàn bộ COCO, đạt khoảng 98% kết quả gốc. Nhóm em đề xuất AdaSLQ, tăng thêm 6–9 điểm. Hướng tiếp theo: tách tập validation, chạy nhiều seed, lọc ảnh trùng, thu nhỏ bộ sinh query, và thử trên dữ liệu tiếng Việt.

---

## Slide 18: Cảm ơn và hỏi đáp
**Thời gian:** 10 giây

> Nhóm em xin cảm ơn cô và các bạn đã lắng nghe. Nhóm em rất mong nhận được câu hỏi và góp ý ạ.

---

## Phân công gợi ý

| Phần | Slide | Người trình bày | Thời gian |
|---|---|---|---|
| Mở đầu, bài toán, động lực | 1–4 | Hoàng Văn Kiên | ~3,5 phút |
| Phương pháp SLQ, KARR-Bench, kết quả bài báo | 5–9 | Nguyễn Thị Quỳnh Trang | ~4,5 phút |
| Tái hiện: thiết lập và kết quả | 10–11 | Đỗ Nhật Minh | ~2,5 phút |
| Cải tiến: bốn hướng và ablation | 12–14 | Nguyễn Minh Quang | ~3,5 phút |
| AdaSLQ, phản biện, kết luận | 15–18 | Trần Khánh Hưng | ~3 phút |

Quang là người chạy các thí nghiệm A100, nên nên trình bày phần cải tiến và trả lời câu hỏi kỹ thuật về phần này.

## Mẹo khi trình bày
- **Slide 5, 10, 12 và 16** là nơi cô dễ hỏi nhất.
- Nói rõ số liệu tái lập và AdaSLQ là **chạy 1 lần**, chọn trên tập test. Nói trước điều này thì tốt hơn để cô phát hiện ra.
- Khi nhắc đến AdaSLQ, luôn nói kèm **10,59M tham số**.
- Mở sẵn file `bao_cao_SLQ.html` trên máy làm phương án dự phòng.
