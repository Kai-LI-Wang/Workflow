SELECT DISTINCT a.*, b.���綵��, b.���絲�G ,  g.��O����W��, g.�ץ���O
FROM [FRBDM2].[dbo].[�������_�g_�����Ƥ��R�j��_������_11] AS a 
LEFT JOIN [��J���~��Ҭd��޲z��T�t��].[dbo].[�D�����i-���~����ץ���] AS b 
ON b.ñ�f�֭�\�i���s�� = a.ñ�f�֭�\�i���s�� 
LEFT JOIN  
(SELECT c.����׸�, c.�f�~�������C, c.�Ͳ���O,e.��O����W��, f.�ץ���O, f.���z��� FROM 
[��J���~��Ҭd��޲z��T�t��].[dbo].[��J���~�d�������] AS c  
LEFT JOIN [��J���~��Ҭd��޲z��T�t��].[dbo].[�Ͳ���O�N�X��] e  ON c.�Ͳ���O = e.��O�N�X 
LEFT JOIN [��J���~��Ҭd��޲z��T�t��].[dbo].[��J���~�d��D��] f ON c.����׸� = f.�ӽЮѸ��X) AS g
ON g.����׸� = a.ñ�f�֭�\�i���s��



 

