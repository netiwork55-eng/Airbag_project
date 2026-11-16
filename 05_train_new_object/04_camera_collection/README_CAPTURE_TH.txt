Airbag camera collection

01_airbag_images_to_label
  รูป Airbag จากกล้องจริง ต้องนำไปตีกรอบ label ก่อน train YOLO
  ห้ามสร้าง label ว่างให้รูปในโฟลเดอร์นี้ เพราะจะสอนผิดว่า Airbag คือ background

02_background_images_no_airbag
  รูปที่ไม่มี Airbag ใช้เป็น negative background ได้
  รูปพวกนี้สามารถสร้าง label ว่างตอนรวม dataset ได้

capture_log.csv
  บันทึกว่ารูปแต่ละใบถ่ายตามโจทย์มุม/แสง/ระยะอะไร

แนะนำขั้นต่ำ:
  ถ่าย Airbag 3-5 รูปต่อ step
  ถ่าย background 30-100 รูปจากพื้นที่จริงที่ใช้งาน
  หลังถ่ายเสร็จให้นำ Airbag images ไป label กรอบแล้ว export เป็น YOLOv8
