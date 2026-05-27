# data/

Thư mục này chứa toàn bộ dữ liệu của dự án, được tách theo trạng thái xử lý để dễ quản lý và tái sử dụng.

## Các thư mục con

- `raw/`: dữ liệu gốc, chưa chỉnh sửa.
- `processed/`: dữ liệu sau làm sạch, mã hóa, chuẩn hóa hoặc tạo đặc trưng.
- `external/`: dữ liệu bổ sung từ nguồn ngoài, nếu có.

## Quy ước

- Không chỉnh sửa trực tiếp dữ liệu gốc trong `raw/`.
- Mỗi bước tiền xử lý nên tạo ra một phiên bản rõ ràng trong `processed/`.
- Nếu thêm dữ liệu ngoài, hãy ghi rõ nguồn và thời gian lấy dữ liệu.