# app/

Thư mục này chứa phần ứng dụng để trình bày kết quả của dự án cho người dùng cuối, thường là dashboard hoặc Streamlit app.

## Mục đích

- Hiển thị dữ liệu đầu vào và các chỉ số chính.
- Cho phép thử dự đoán trên mẫu mới.
- Trình bày biểu đồ, so sánh mô hình và phần giải thích kết quả.

## Nội dung dự kiến

- `streamlit_app.py`: điểm vào chính của ứng dụng.
- `assets/`: hình ảnh, CSS, icon hoặc file tĩnh khác.

## Lưu ý

Chỉ nên đưa vào đây các file phục vụ trình bày hoặc demo. Logic xử lý dữ liệu và huấn luyện nên để trong `src/` hoặc notebook.