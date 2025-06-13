import pandas as pd
import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

# --- 폰트 오류 해결 ....------------------
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
# --------------------------------------------


# .sas7bdat 파일 경로
file_path = r'C:\Users\minse\Downloads\HN23_ALL\hn23_all.sas7bdat'

# ------------------오류 해결2 (gpt)---------------------------------------------
# --- pandas 출력 옵션 설정 (모든 컬럼이 보이도록) ---
# 기존 옵션 값을 저장하여 스크립트 종료 시 복원 (권장)
original_max_columns = pd.get_option('display.max_columns')
original_width = pd.get_option('display.width')
original_unicode_east_asian_width = pd.get_option('display.unicode.east_asian_width')
original_max_rows = pd.get_option('display.max_rows')
original_max_colwidth = pd.get_option('display.max_colwidth')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.unicode.east_asian_width', True)  # 한글 컬럼명 정렬을 위해 유지
pd.set_option('display.max_colwidth', None)  # 컬럼 내용이 잘리지 않도록 설정
# ------------------------------------------------------------------------------------




######---------------- 필요한 데이터만 뽑아 데이터프레임 재구성------------------------------------------------------------------------
try:
    # 파일 경로 존재 여부 먼저 확인
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"'{file_path}' 파일을 찾을 수 없습니다. 경로를 다시 확인해주세요.")

    # SAS 파일을 데이터프레임으로 읽기
    df = pd.read_sas(file_path, encoding='latin-1')
    print(f"'{file_path}' 파일을 성공적으로 읽었습니다.")

    # 사용할 필수 기본 정보 컬럼 목록 (가장 왼쪽에 위치)
    base_columns = ['ID', 'sex', 'age']

    # 사용할 건강지수지표 컬럼 목록
    health_columns = ['HE_TG', 'HE_chol', 'HE_BMI', 'HE_LDL_drct', 'HE_HDL_st2', 'HE_glu', 'HE_sbp', 'HE_dbp']

    # 모든 필수 컬럼들을 합쳐 순서대로 정렬
    required_columns = base_columns + health_columns

    # 컬럼명 매핑 딕셔너리 (영문: 한글)
    column_name_mapping = {
        'HE_TG': '중성지방',
        'HE_chol': '총 콜레스테롤',
        'HE_LDL_drct': '저밀도지단백콜레스테롤',
        'HE_HDL_st2': '고밀도지단백콜레스테롤',
        'HE_glu': '공복혈당',
        'HE_sbp': '최종 수축기 혈압',
        'HE_dbp': '최종 이완기혈압',
        'HE_BMI': '체질량지수'
    }

    # 필요한 컬럼들이 데이터프레임에 모두 존재하는지 확인
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"경고: 다음 컬럼들이 데이터프레임에 존재하지 않습니다: {missing_columns}")
        existing_columns = [col for col in required_columns if col in df.columns]
        if not existing_columns:
            raise ValueError("요청된 컬럼 중 유효한 컬럼이 하나도 없습니다.")
        print(f"존재하는 컬럼들로 데이터프레임을 생성합니다: {existing_columns}")
        selected_df = df[existing_columns]
    else:
        print(f"필요한 모든 컬럼 {required_columns}이(가) 데이터프레임에 존재합니다.")
        selected_df = df[required_columns]

    # 컬럼명 변경
    rename_map_for_selected_df = {eng_name: kor_name for eng_name, kor_name in column_name_mapping.items() if
                                  eng_name in selected_df.columns}
    selected_df = selected_df.rename(columns=rename_map_for_selected_df)

    # --- 65세 이상 노인 데이터로 필터링 추가 ---
    if 'age' in selected_df.columns:
        initial_row_count = len(selected_df)
        selected_df = selected_df[selected_df['age'] >= 65].copy()
        print(f"\n--- 65세 이상 노인 데이터로 필터링 완료 ---")
        print(f"원본 데이터 {initial_row_count} 행에서 65세 이상 {len(selected_df)} 행으로 필터링되었습니다.")
    else:
        print("\n경고: '나이' 컬럼이 없어 65세 이상 필터링을 건너뜁니다.")

    print("\n--- 선택된 컬럼들로 구성된 요약 데이터프레임 (상위 5개 행, 모든 컬럼 한글명으로 표시) ---")
    print(selected_df.head(5))
    # print(selected_df)

    # print("\n--- 선택된 컬럼들로 구성된 요약 데이터프레임 정보 (df.info()) ---")
    # selected_df.info()

    # print("\n--- 선택된 컬럼들로 구성된 데이터프레임 컬럼 목록 (한글명) ---")
    # print(selected_df.columns.tolist())


    #-------------------df_senior 데이터프레임----------------------------#
    # 추가로 전체 건강상태 데이터에서 65세 이상만 뽑아냄 (열 선택없이) (sklearn학습용)
    # 'age' 컬럼이 65세 이상인 데이터만 필터링하여 새 DataFrame 생성
    if 'age' in df.columns:
        df_senior = df[df['age'] >= 65]
        print(f"\n--- 65세 이상으로 필터링된 데이터프레임 (총 {len(df_senior)}개 행) ---")
        print(df_senior.head(5))
    else:
        print("\n경고: 'age' 컬럼이 데이터프레임에 없어 65세 이상 필터링을 수행할 수 없습니다.")
    #-----------------------------------------------------------------





    ##########################################################################################################################################
    #----------------1. 고혈압 위험군 (수축기혈압 120 이상, 이완기혈압 80이상)
    hypertension_risk_ids = []
    if '최종 수축기 혈압' in selected_df.columns and '최종 이완기혈압' in selected_df.columns:
        hypertension_risk_group_df = selected_df[
            (selected_df['최종 수축기 혈압'].notna()) &
            (selected_df['최종 이완기혈압'].notna()) &
            (selected_df['최종 수축기 혈압'] >= 120) &
            (selected_df['최종 이완기혈압'] >= 80)
            ].copy()

        if not hypertension_risk_group_df.empty:
            hypertension_risk_ids = hypertension_risk_group_df['ID'].tolist()
            print(f"\n- 1.고혈압 위험군 ({len(hypertension_risk_ids)}명) ID 리스트 (상위 5개) ---")
            print(hypertension_risk_ids[:5])
        else:
            print("고혈압 위험군에 해당하는 데이터가 없습니다.")
    else:
        print("고혈압 위험군을 식별하는 데 필요한 '최종 수축기 혈압' 또는 '최종 이완기혈압' 컬럼이 없습니다.")
    # ------------------------------------------------------------



    ###--------------------------- 2. 비만 위험군 (BMI 25 이상)------------------
    obesity_risk_ids = []
    if '체질량지수' in selected_df.columns:
        obesity_risk_df = selected_df[
            (selected_df['체질량지수'].notna()) & (selected_df['체질량지수'] >= 25)
            ].copy()
        obesity_risk_ids = obesity_risk_df['ID'].tolist()
        print(f"\n- 2.비만 위험군 ({len(obesity_risk_ids)}명) ID 리스트 (상위 5개):")
        print(obesity_risk_ids[:5])
    else:
        print("비만 위험군을 식별하는 데 필요한 '체질량지수' 컬럼이 없습니다.")
    #----------------------------------------------------------------------------


    ###-------------------- 3. 당뇨병 위험군 (공복혈당 126 이상)-------------------------
    diabetes_risk_ids = []
    if '공복혈당' in selected_df.columns:
        diabetes_risk_df = selected_df[
            (selected_df['공복혈당'].notna()) & (selected_df['공복혈당'] >= 126)
            ].copy()
        diabetes_risk_ids = diabetes_risk_df['ID'].tolist()
        print(f"\n- 3.당뇨 위험군 ({len(diabetes_risk_ids)}명) ID 리스트 (상위 5개):")
        print(diabetes_risk_ids[:5])
    else:
        print("당뇨 위험군을 식별하는 데 필요한 '공복혈당' 컬럼이 없습니다.")
    #--------------------------------------------------------------------------------



    ###----------------- 4. 고지질혈증 위험군  (중성지방 ≥ 200 and 총 콜레스테롤 ≥ 240)----------------------------
    risk_ids = []

    # 필요한 컬럼이 모두 존재할 때만 처리
    if '중성지방' in selected_df.columns and '총 콜레스테롤' in selected_df.columns:

        combined_risk_df = selected_df[
            (selected_df['중성지방'].notna()) &
            (selected_df['총 콜레스테롤'].notna()) &
            (selected_df['중성지방'] >= 200) &
            (selected_df['총 콜레스테롤'] >= 240)
            ].copy()

        risk_ids = combined_risk_df['ID'].tolist()

        print(f"\n- 4.고지질혈증 위험군 ({len(risk_ids)}명) ID 리스트 (상위 5개):")
        print(risk_ids[:5])

    else:
        print("중성지방 및/또는 총 콜레스테롤 컬럼이 없습니다.")
    #----------------------------------------------------------------------------------
    ##############################################################################################

    # ---------------- 질병 여부 열 추가  ----------------
    selected_df['고혈압'] = selected_df['ID'].apply(lambda x: 1 if x in hypertension_risk_ids else 0)
    selected_df['비만'] = selected_df['ID'].apply(lambda x: 1 if x in obesity_risk_ids else 0)
    selected_df['당뇨병'] = selected_df['ID'].apply(lambda x: 1 if x in diabetes_risk_ids else 0)
    selected_df['이상지질혈증'] = selected_df['ID'].apply(lambda x: 1 if x in risk_ids else 0)

    # ---------------- 위험도 level 분류 ----------------
    selected_df['질병_개수'] = selected_df[['고혈압', '비만', '당뇨병', '이상지질혈증']].sum(axis=1)


    # 위험도 level 할당
    def assign_level(count):
        if count == 1:
            return 1
        elif count == 2:
            return 2
        elif count == 3:
            return 3
        elif count == 4:
            return 4
        else:
            return 0


    selected_df['level'] = selected_df['질병_개수'].apply(assign_level)
    # 결과 확인
    print("\n--- 최종 데이터프레임 (상위 5개 행) ---")
    print(selected_df[['ID', '고혈압', '비만', '당뇨병', '이상지질혈증', '질병_개수', 'level']].head(5))






    ###---------------- 질병 보유자 수 시각화 ----------------
    disease_counts = {
        '고혈압': selected_df['고혈압'].sum(),
        '비만': selected_df['비만'].sum(),
        '당뇨병': selected_df['당뇨병'].sum(),
        '이상지질혈증': selected_df['이상지질혈증'].sum()
    }

    plt.figure(figsize=(8, 5))
    plt.bar(disease_counts.keys(), disease_counts.values())
    plt.title('질병별 보유자 수')
    plt.xlabel('질병')
    plt.ylabel('환자 수')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()




except FileNotFoundError as e:  # --- except 블록 시작 ---
    print(f"오류: {e}")
except Exception as e:  # 다른 모든 예외를 처리
    print(f"파일을 읽거나 처리하는 중 오류가 발생했습니다: {e}")

finally:  # --- finally 블록 시작 ---
    # --- 출력 옵션 초기화 ---
    pd.set_option('display.max_columns', original_max_columns)
    pd.set_option('display.width', original_width)
    pd.set_option('display.unicode.east_asian_width', original_unicode_east_asian_width)
    pd.set_option('display.max_rows', original_max_rows)
    pd.set_option('display.max_colwidth', original_max_colwidth)
    print("\n--- Pandas 출력 옵션이 초기화되었습니다. ---")





# ---------------- 사용자 정의 함수: ID로 질병 및 level 확인 ----------------
def show_health_status(user_id, df):
    row = df[df['ID'] == user_id]

    if row.empty:
        print(f" ID '{user_id}' 에 해당하는 데이터가 없습니다.")
        return

    diseases = ['고혈압', '비만', '당뇨병', '이상지질혈증']
    has_disease = []

    for d in diseases:
        if row.iloc[0][d] == 1:
            has_disease.append(d)

    if has_disease:
        print("-------------------------------------------")
        print("**보유 질병:", ', '.join(has_disease))
    else:
        print("-------------------------------------------")
        print("** 보유 질병: 없음")

    print(f"**심장질환 위험도 (level)**: {row.iloc[0]['level']}단계")
    print("-------------------------------------------")


# ---------------- 사용자 입력 루프 ----------------
while True:
    user_input = input("\n확인할 사용자 ID를 입력하세요 (종료하려면 'exit' 또는 '종료' 입력): ").strip()

    if user_input.lower() in ['exit', '종료']:
        print("프로그램을 종료합니다.")
        break

    show_health_status(user_input, selected_df)

