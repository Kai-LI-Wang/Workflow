
DROP TABLE IF EXISTS #t2
SELECT * INTO #t2 FROM  
(SELECT DISTINCT a.*, b.���綵��, b.���絲�G ,  g.��O����W��, g.�ץ���O
FROM [FRBDM2].[dbo].[�������_�g_�����Ƥ��R�j��_������_11] AS a 
LEFT JOIN [��J���~��Ҭd��޲z��T�t��].[dbo].[�D�����i-���~����ץ���] AS b 
ON b.ñ�f�֭�\�i���s�� = a.ñ�f�֭�\�i���s�� 
LEFT JOIN  
(SELECT c.����׸�, c.�f�~�������C, c.�Ͳ���O,e.��O����W��, f.�ץ���O, f.���z��� FROM 
[��J���~��Ҭd��޲z��T�t��].[dbo].[��J���~�d�������] AS c  
LEFT JOIN [��J���~��Ҭd��޲z��T�t��].[dbo].[�Ͳ���O�N�X��] e  ON c.�Ͳ���O = e.��O�N�X 
LEFT JOIN [��J���~��Ҭd��޲z��T�t��].[dbo].[��J���~�d��D��] f ON c.����׸� = f.�ӽЮѸ��X) AS g
ON g.����׸� = a.ñ�f�֭�\�i���s�� 
WHERE b.���綵�� IS NOT NULL ) AS k



DROP TABLE IF EXISTS #t3
SELECT * INTO #t3 FROM (
SELECT DISTINCT E.*, 
				SUM(E.���絲�G_���s) OVER (PARTITION BY E.ñ�f�֭�\�i���s�� ORDER BY ñ�f�֭�\�i���s��) AS SumByInspection
FROM 
	(
	SELECT *,
	CASE 
		WHEN ���綵�� LIKE '%[-]%[��l|�~�[]%' THEN NULL 
		ELSE ���綵��
	END AS ���綵��_�ư���l�~�[,
	CASE 
		WHEN ���絲�G LIKE 'Y' THEN 0
		ELSE 1
	END AS ���絲�G_���s
	FROM #t2 WHERE �ץ���O IS NULL AND ���綵�� IS NOT NULL 
	) AS E 
WHERE  E.���綵��_�ư���l�~�[ IS NOT NULL ) as f

ALTER TABLE #t3
DROP COLUMN ���絲�G_���s, ���絲�G, ���涵��, ���綵��, ���綵��_�ư���l�~�[, ���礣�X��


SELECT DISTINCT *, LAST_VALUE(IIF(SumByInspection > 0 , '���X��', '�X��')) OVER (PARTITION BY ñ�f�֭�\�i���s�� ORDER BY ñ�f�֭�\�i���s��) AS ���絲�G  FROM #t3

