import pandas as pd
import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt 
# .sas7bdat 파일 경로
file_path = r'C:\Users\minse\Downloads\HN23_ALL\hn23_all.sas7bdat'

# --- pandas 출력 옵션 설정 (모든 컬럼이 보이도록) ---
# 기존 옵션 값을 저장하여 스크립트 종료 시 복원 (권장)
original_max_columns = pd.get_option('display.max_columns')
original_width = pd.get_option('display.width')
original_unicode_east_asian_width = pd.get_option('display.unicode.east_asian_width')
original_max_rows = pd.get_option('display.max_rows')
original_max_colwidth = pd.get_option('display.max_colwidth')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.unicode.east_asian_width', True) # 한글 컬럼명 정렬을 위해 유지
pd.set_option('display.max_colwidth', None) # 컬럼 내용이 잘리지 않도록 설정


# --- classify_risk_levels 함수 정의 ---
def classify_risk_levels(
    hypertension_risk_ids: list,
    obesity_risk_ids: list,
    glucose_risk_ids: list,
    diabetes_risk_ids: list,
    tg_risk_ids: list,
    cholesterol_risk_ids: list
) -> dict:
    """
    주어진 6가지 질병 위험군 ID 리스트를 기반으로 각 ID의 위험도를 6단계로 분류합니다.
    """
    all_risk_ids = (
        hypertension_risk_ids +
        obesity_risk_ids +
        glucose_risk_ids +
        diabetes_risk_ids +
        tg_risk_ids +
        cholesterol_risk_ids
    )

    id_risk_counts = Counter(all_risk_ids)

    classified_risks = {
        '초고도 위험군': [],  # 6개 포함
        '고도 위험군': [],    # 5개 포함
        '상당 위험군': [],   # 4개 포함
        '중등도 위험군': [], # 3개 포함
        '주의필요': [],      # 2개 포함
        '관심요망': []       # 1개 포함
    }

    for individual_id, count in id_risk_counts.items():
        if count == 6:
            classified_risks['초고도 위험군'].append(individual_id)
        elif count == 5:
            classified_risks['고도 위험군'].append(individual_id)
        elif count == 4:
            classified_risks['상당 위험군'].append(individual_id)
        elif count == 3:
            classified_risks['중등도 위험군'].append(individual_id)
        elif count == 2:
            classified_risks['주의필요'].append(individual_id)
        elif count == 1:
            classified_risks['관심요망'].append(individual_id)

    for risk_level in classified_risks:
        classified_risks[risk_level].sort()

    return classified_risks
# --- classify_risk_levels 함수 정의 끝 ---


# --- 나의 위험군 분석 함수 정의 (수정됨) ---
def analyze_my_risk(
    my_id: str, # 분석할 개인의 ID
    hypertension_risk_ids: list,
    obesity_risk_ids: list,
    glucose_risk_ids: list,
    diabetes_risk_ids: list,
    tg_risk_ids: list,
    cholesterol_risk_ids: list
) -> None:
    """
    특정 개인 ID가 포함되는 질병 위험군과 해당 ID의 종합적인 위험군도를 출력합니다.
    """
    print(f"\n--- ID '{my_id}' 님의 건강 위험도 분석 결과 ---")

    # ID가 포함된 개별 위험군 확인
    involved_risks = []
    # 위험군 이름과 해당 ID 리스트를 튜플로 묶어 관리 (고혈압, 비만, 혈당, 당뇨, 중성지방, 콜레스테롤)
    risk_categories = [
        ('고혈압', hypertension_risk_ids),
        ('비만', obesity_risk_ids),
        ('혈당', glucose_risk_ids),
        ('당뇨', diabetes_risk_ids),
        ('중성지방', tg_risk_ids),
        ('콜레스테롤', cholesterol_risk_ids)
    ]

    for risk_name, risk_list in risk_categories:
        if my_id in risk_list:
            involved_risks.append(risk_name) # '고혈압 위험군' 대신 '고혈압'만 추가

    if involved_risks:
        # 콤마로 구분하여 한 줄로 출력
        print(f"'{my_id}' 님은 **{', '.join(involved_risks)}** 수치가 높아 위험군에 속합니다.")
    else:
        print(f"'{my_id}' 님은 현재 6가지 주요 질병 위험군 중 어느 곳에도 포함되지 않습니다. (건강한 편)")

    # 종합적인 위험군도 판단 (이 부분은 이전과 동일합니다)
    num_risks = len(involved_risks)
    overall_risk_level = "해당 없음 (데이터 부족 또는 모든 위험군 제외)"

    if num_risks == 6:
        overall_risk_level = '초고도 위험군'
    elif num_risks == 5:
        overall_risk_level = '고도 위험군'
    elif num_risks == 4:
        overall_risk_level = '상당 위험군'
    elif num_risks == 3:
        overall_risk_level = '중등도 위험군'
    elif num_risks == 2:
        overall_risk_level = '주의필요'
    elif num_risks == 1:
        overall_risk_level = '관심요망'
    elif num_risks == 0:
        overall_risk_level = '매우 양호' # 0개 포함 시의 분류

    print(f"\n'{my_id}' 님의 종합적인 위험군도는 >>{overall_risk_level}<< 에 해당합니다.")
    print("------------------------------------------")




try: # --- try 블록 시작 ---
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
    rename_map_for_selected_df = {eng_name: kor_name for eng_name, kor_name in column_name_mapping.items() if eng_name in selected_df.columns}
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
    #print(selected_df.head(5))
    print(selected_df)

    print("\n--- 선택된 컬럼들로 구성된 요약 데이터프레임 정보 (df.info()) ---")
    selected_df.info()

    print("\n--- 선택된 컬럼들로 구성된 데이터프레임 컬럼 목록 (한글명) ---")
    print(selected_df.columns.tolist())

    ###################### 1. 고혈압 위험군 (수축기혈압 120 이상, 이완기혈압 80이상)
    print("\n--- 고혈압 위험군 데이터 추출 ---")
    hypertension_risk_ids = []
    if '최종 수축기 혈압' in selected_df.columns and '최종 이완기혈압' in selected_df.columns:
        hypertension_risk_group_df = selected_df[
            (selected_df['최종 수축기 혈압'].notna()) &
            (selected_df['최종 이완기혈압'].notna()) &
            (selected_df['최종 수축기 혈압'] >= 120) &
            (selected_df['최종 이완기혈압'] >= 80)
        ].copy()

        if not hypertension_risk_group_df.empty:
            print(f"고혈압 위험군을 {len(hypertension_risk_group_df)}명 찾았습니다.")
            print("\n--- 고혈압 위험군 데이터프레임 (상위 5개 행) ---")
            print(hypertension_risk_group_df.head(5))
            hypertension_risk_ids = hypertension_risk_group_df['개인표본조사구'].tolist()
            print(f"\n--- 고혈압 위험군 ID 리스트 (상위 5개) ---")
            print(hypertension_risk_ids[:5])
        else:
            print("고혈압 위험군에 해당하는 데이터가 없습니다.")
    else:
        print("고혈압 위험군을 식별하는 데 필요한 '최종 수축기 혈압' 또는 '최종 이완기혈압' 컬럼이 없습니다.")
    # --- 고혈압 위험군 필터링 및 저장 끝 ---


    # --- 각 지표별 위험군 필터링 및 ID 리스트 저장 ---
    print("\n--- 각 지표별 위험군 데이터 추출 및 리스트 저장 ---")

    ################ 2. 비만 위험군 (BMI 23 이상)
    obesity_risk_ids = []
    if '체질량지수' in selected_df.columns:
        obesity_risk_df = selected_df[
            (selected_df['체질량지수'].notna()) & (selected_df['체질량지수'] >= 23)
        ].copy()
        obesity_risk_ids = obesity_risk_df['개인표본조사구'].tolist()
        print(f"\n- 비만 위험군 ({len(obesity_risk_ids)}명) ID 리스트 (상위 5개):")
        print(obesity_risk_ids[:5])
    else:
        print("비만 위험군을 식별하는 데 필요한 '체질량지수' 컬럼이 없습니다.")

    ######################### 3. 혈당 위험군 (공복혈당 100 이상)
    glucose_risk_ids = []
    if '공복혈당' in selected_df.columns:
        glucose_risk_df = selected_df[
            (selected_df['공복혈당'].notna()) & (selected_df['공복혈당'] >= 100)
        ].copy()
        glucose_risk_ids = glucose_risk_df['개인표본조사구'].tolist()
        print(f"\n- 혈당 위험군 ({len(glucose_risk_ids)}명) ID 리스트 (상위 5개):")
        print(glucose_risk_ids[:5])
    else:
        print("혈당 위험군을 식별하는 데 필요한 '공복혈당' 컬럼이 없습니다.")

    ##################### 4. 당뇨 위험군 (공복혈당 126 이상)
    diabetes_risk_ids = []
    if '공복혈당' in selected_df.columns:
        diabetes_risk_df = selected_df[
            (selected_df['공복혈당'].notna()) & (selected_df['공복혈당'] >= 126)
        ].copy()
        diabetes_risk_ids = diabetes_risk_df['개인표본조사구'].tolist()
        print(f"\n- 당뇨 위험군 ({len(diabetes_risk_ids)}명) ID 리스트 (상위 5개):")
        print(diabetes_risk_ids[:5])
    else:
        print("당뇨 위험군을 식별하는 데 필요한 '공복혈당' 컬럼이 없습니다.")

    ################## 5. 중성지방 위험군 (중성지방 150 이상)
    tg_risk_ids = []
    if '중성지방' in selected_df.columns:
        tg_risk_df = selected_df[
            (selected_df['중성지방'].notna()) & (selected_df['중성지방'] >= 150)
        ].copy()
        tg_risk_ids = tg_risk_df['개인표본조사구'].tolist()
        print(f"\n- 중성지방 위험군 ({len(tg_risk_ids)}명) ID 리스트 (상위 5개):")
        print(tg_risk_ids[:5])
    else:
        print("중성지방 위험군을 식별하는 데 필요한 '중성지방' 컬럼이 없습니다.")

    ################### 6. 콜레스테롤 위험군 (총 콜레스테롤 200 이상)
    cholesterol_risk_ids = []
    if '총 콜레스테롤' in selected_df.columns:
        cholesterol_risk_df = selected_df[
            (selected_df['총 콜레스테롤'].notna()) & (selected_df['총 콜레스테롤'] >= 200)
        ].copy()
        cholesterol_risk_ids = cholesterol_risk_df['개인표본조사구'].tolist()
        print(f"\n- 콜레스테롤 위험군 ({len(cholesterol_risk_ids)}명) ID 리스트 (상위 5개):")
        print(cholesterol_risk_ids[:5])
    else:
        print("콜레스테롤 위험군을 식별하는 데 필요한 '총 콜레스테롤' 컬럼이 없습니다.")
    # --- 각 지표별 위험군 필터링 및 리스트 저장 끝 ---




    # --- 6가지 주요 기저질환별 환자 수 시각화 (추가된 부분) ---
    print("\n\n############################################")
    print("### 6가지 주요 기저질환별 환자 수 시각화 ###")
    print("############################################")

    # 각 위험군별 환자 수 계산
    risk_category_counts = {
        '고혈압': len(hypertension_risk_ids),
        '비만': len(obesity_risk_ids),
        '혈당': len(glucose_risk_ids),
        '당뇨': len(diabetes_risk_ids),
        '중성지방': len(tg_risk_ids),
        '콜레스테롤': len(cholesterol_risk_ids)
    }

    # x축 레이블과 y축 값 준비
    categories = list(risk_category_counts.keys())
    counts = list(risk_category_counts.values())

    # 막대 그래프 생성
    plt.figure(figsize=(12, 7)) # 그래프 크기 설정
    bars = plt.bar(categories, counts, color='skyblue')

    # 그래프 제목 및 축 레이블 설정
    plt.title('6가지 주요 기저질환별 65세 이상 노인 환자 수', fontsize=16)
    plt.xlabel('기저질환 종류', fontsize=14)
    plt.ylabel('환자 수', fontsize=14)
    plt.xticks(fontsize=12, rotation=45, ha='right') # x축 레이블 기울기 및 정렬
    plt.yticks(fontsize=12)

    # 각 막대 위에 환자 수 표시
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, round(yval), ha='center', va='bottom', fontsize=10)


    plt.grid(axis='y', linestyle='--', alpha=0.7) # y축 그리드 라인 추가
    plt.tight_layout() # 레이블이 잘리지 않도록 레이아웃 조정
    plt.show() # 그래프 표시

    print("--- 6가지 주요 기저질환별 환자 수 시각화 완료 ---")






    # --- 다중 위험군 분류 시작 ---
    print("\n--- 다중 위험군 분류 시작 ---")
    if ('hypertension_risk_ids' in locals() and 'obesity_risk_ids' in locals() and
        'glucose_risk_ids' in locals() and 'diabetes_risk_ids' in locals() and
        'tg_risk_ids' in locals() and 'cholesterol_risk_ids' in locals()):
        classified_risk_results = classify_risk_levels(
            hypertension_risk_ids,
            obesity_risk_ids,
            glucose_risk_ids,
            diabetes_risk_ids,
            tg_risk_ids,
            cholesterol_risk_ids
        )

        for risk_level, ids in classified_risk_results.items():
            print(f"\n{risk_level} ({len(ids)}명):")
            if ids:
                print(ids[:10]) # 상위 10개 ID 출력
            else:
                print("해당하는 ID 없음.")

        print("\n--- 다중 위험군 분류 완료 ---")
    else:
        print("\n경고: 모든 위험군 ID 리스트가 생성되지 않아 다중 위험군 분류를 건너뜁니다.")

    # --- 나의 ID로 위험군 분석 실행 ---
    print("\n\n############################################")
    print("### 개인 ID 기반 위험도 분석 시작 ###")
    print("############################################")

    # 사용자로부터 ID 입력 받기
    user_id_input = input("분석할 개인의 고유 ID를 입력해주세요 (예: 'A000001'): ").strip()

    # 입력된 ID가 전체 데이터프레임에 존재하는지 확인
    if '개인표본조사구' in selected_df.columns and user_id_input in selected_df['개인표본조사구'].values:
        analyze_my_risk(
            user_id_input,
            hypertension_risk_ids,
            obesity_risk_ids,
            glucose_risk_ids,
            diabetes_risk_ids,
            tg_risk_ids,
            cholesterol_risk_ids
        )
    else:
        print(f"\n오류: 입력하신 ID '{user_id_input}'는 데이터에 존재하지 않거나 '개인표본조사구' 컬럼이 없습니다.")
        print("정확한 ID를 입력했는지 확인해주세요.")


except FileNotFoundError as e: # --- except 블록 시작 ---
    print(f"오류: {e}")
except Exception as e: # 다른 모든 예외를 처리
    print(f"파일을 읽거나 처리하는 중 오류가 발생했습니다: {e}")

finally: # --- finally 블록 시작 ---
    # --- 출력 옵션 초기화 ---
    pd.set_option('display.max_columns', original_max_columns)
    pd.set_option('display.width', original_width)
    pd.set_option('display.unicode.east_asian_width', original_unicode_east_asian_width)
    pd.set_option('display.max_rows', original_max_rows)
    pd.set_option('display.max_colwidth', original_max_colwidth)
    print("\n--- Pandas 출력 옵션이 초기화되었습니다. ---")
