| Sno  | Layer-Name   | Input | Kernel | Padding | Stride | J_in   | Output  | RF      |
| ---- | ------------ | ----- | ------ | ------- | ------ | ------ | ------- | ------- |
| 1    | C1           | 224   | 7      | 3       | 2      | **1**  | **112** | **7**   |
| 2    | MP1          | 112   | 3      | 0       | 2      | **2**  | **56**  | **11**  |
| 3    | C2           | 56    | 3      | 1       | 1      | **4**  | **56**  | **19**  |
| 4    | MP2          | 56    | 3      | 0       | 2      | **4**  | **28**  | **27**  |
| 5    | 3A_Inception | 28    | 5      | 2       | 1      | **8**  | **28**  | **59**  |
| 6    | 3B_Inception | 28    | 5      | 2       | 1      | **8**  | **28**  | **91**  |
| 7    | MP3          | 28    | 3      | 0       | 2      | **8**  | **14**  | **107** |
| 8    | 4A_Inception | 14    | 5      | 2       | 1      | **16** | **14**  | **171** |
| 9    | 4B_Inception | 14    | 5      | 2       | 1      | **16** | **14**  | **235** |
| 10   | 4C_Inception | 14    | 5      | 2       | 1      | **16** | **14**  | **299** |
| 11   | 4D_Inception | 14    | 5      | 2       | 1      | **16** | **14**  | **363** |
| 12   | 4E_Inception | 14    | 5      | 2       | 1      | **16** | **14**  | **427** |
| 13   | MP4          | 14    | 3      | 0       | 2      | **16** | **7**   | **459** |
| 14   | 5A_Inception | 7     | 5      | 2       | 1      | **32** | **7**   | **587** |
| 15   | 5B_Inception | 7     | 5      | 2       | 1      | **32** | **7**   | **715** |
| 16   | AvgPool      | 7     | 7      | 0       | 1      | **32** | **1**   | **907** |
|      |              |       |        |         |        |        |         |         |