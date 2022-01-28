
DROP TABLE IF EXISTS #t2
SELECT * INTO #t2 FROM  
(SELECT DISTINCT a.*, b.檢驗項目, b.檢驗結果 ,  g.國別中文名稱, g.案件註記
FROM [FRBDM2].[dbo].[邊境檢驗_週_香辛料分析大表_全報單_11] AS a 
LEFT JOIN [輸入食品邊境查驗管理資訊系統].[dbo].[非報不可-食品檢驗案件檔] AS b 
ON b.簽審核准許可文件編號 = a.簽審核准許可文件編號 
LEFT JOIN  
(SELECT c.報驗案號, c.貨品分類號列, c.生產國別,e.國別中文名稱, f.案件註記, f.受理日期 FROM 
[輸入食品邊境查驗管理資訊系統].[dbo].[輸入食品查驗明細檔] AS c  
LEFT JOIN [輸入食品邊境查驗管理資訊系統].[dbo].[生產國別代碼檔] e  ON c.生產國別 = e.國別代碼 
LEFT JOIN [輸入食品邊境查驗管理資訊系統].[dbo].[輸入食品查驗主檔] f ON c.報驗案號 = f.申請書號碼) AS g
ON g.報驗案號 = a.簽審核准許可文件編號 
WHERE b.檢驗項目 IS NOT NULL ) AS k



DROP TABLE IF EXISTS #t3
SELECT * INTO #t3 FROM (
SELECT DISTINCT E.*, 
				SUM(E.檢驗結果_重製) OVER (PARTITION BY E.簽審核准許可文件編號 ORDER BY 簽審核准許可文件編號) AS SumByInspection
FROM 
	(
	SELECT *,
	CASE 
		WHEN 檢驗項目 LIKE '%[-]%[原子|外觀]%' THEN NULL 
		ELSE 檢驗項目
	END AS 檢驗項目_排除原子外觀,
	CASE 
		WHEN 檢驗結果 LIKE 'Y' THEN 0
		ELSE 1
	END AS 檢驗結果_重製
	FROM #t2 WHERE 案件註記 IS NULL AND 檢驗項目 IS NOT NULL 
	) AS E 
WHERE  E.檢驗項目_排除原子外觀 IS NOT NULL ) as f

ALTER TABLE #t3
DROP COLUMN 檢驗結果_重製, 檢驗結果, 報單項次, 檢驗項目, 檢驗項目_排除原子外觀, 檢驗不合格


SELECT DISTINCT *, LAST_VALUE(IIF(SumByInspection > 0 , '不合格', '合格')) OVER (PARTITION BY 簽審核准許可文件編號 ORDER BY 簽審核准許可文件編號) AS 檢驗結果  FROM #t3

