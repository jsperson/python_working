SELECT
	inv.contract_type AS Contract_Type,
	inv.contract_number AS Contract_Number,
	inv.contract_line_number AS contract_line_number,
	inv.release_number AS Release_number,
	nvl(case when inv.contract_type = 'SALE' then max(inv.actual_shipment_dt_id) when inv.contract_type = 'PURCHASE' then max(inv.received_dt_id) else 0 end,0) AS Actualization_dt_id,
	cast(nvl(sum(inv.actual_quantity_32e),0) AS NUMERIC(18,10)) AS total_Quantity_32e,
	nvl(max(curr_contract_price_usd_32e),0)  AS total_contract_price_usd_32e,
	CAST((nvl(sum(mtm.exp_settlement_amt_contract),0)*nvl(cast(sum(inv.actual_quantity_32e) as NUMERIC(28,10)),0))/decode(sum(mtm.curr_quantity_32e),0,1,null,1,sum(mtm.curr_quantity_32e)) AS NUMERIC(28,10)) AS total_settlement_amt_32e,
	CAST(nvl(SUM(inv.actual_quantity) ,0) AS NUMERIC(18,10)) AS total_quantity_contract_uom,
	nvl(max(curr_contract_price_contract),0) AS total_contract_price_contract_uom,
	CAST((nvl(sum(mtm.exp_settlement_amt_contract),0)*nvl(cast(sum(inv.actual_quantity) as NUMERIC(28,10)),0))/decode(sum(mtm.curr_quantity_contract),0,1,null,1,sum(mtm.curr_quantity_contract)) AS NUMERIC(28,10)) AS total_settlement_amt_contract_uom,
	max(product_type) AS product_type,
	max(prod_name) AS product_name,
	max(ebs_item) AS EBS_item,
	max(currency_uom_contract) AS contract_currency,
	max(inv.volume_uom_contract) AS contract_uom,
	max(nvl(inv.accounting_dt_id,0)) AS accounting_dt_id,
	max(inv.last_update_date) AS last_update_date,
	NULL AS invoice_unit_price,
	max(inv.invoiced_status) AS invoiced_staus,
	NULL AS accounted_flg,
	mtm.contract_line_id AS contract_line_id,
	max(mtm.inventory_org_id) AS inventory_org_id,
	max(mtm.product_id) AS product_id,
	max(mtm.mele_id) AS mele_id,
	max(mtm.legal_entity_id) AS legal_entity_id,
	greatest(max(inv.last_update_date), max(mtm.update_dt)) AS change_date,
	nvl(CAST(TO_CHAR(TO_DATE(greatest(max(inv.last_update_date), max(mtm.update_dt)),'YYYY-MM-DD'),'YYYYMMDD') AS INT),0)  AS change_dt_id, 
	sysdate AS update_dt,
	sysdate AS insert_dt,
	inv.Contract_Type||'|#|'||inv.Contract_Number||'|#|'||inv.contract_line_number||'|#|'||inv.Release_number AS alternate_key,
	inv.Contract_Type||'|#|'||inv.Contract_Number||'|#|'||inv.contract_line_number||'|#|'||inv.Release_number||'|#|'||change_dt_id AS natural_key,
	--'HIST LOAD' AS blackline_status,
    max(inv.price_status) as price_status,
	max(mele.mele_code) as mele_code,
	max(mele.mele_name) as mele_name,
	max(org.org_code) as inv_org_code,
	max(org.org_name) as inv_org_name
FROM
	ads.rs_mtm_f mtm,
	(
	SELECT
		inv_f.contract_type ,
		inv_F.contract_number ,
		inv_f.contract_line_number ,
		inv_f.release_number ,
		inv_f.source_system_id, 
		inv_f.contract_line_id1 ,
		inv_f.location_id1 ,
		inv_f.product_id1 ,
		inv_f.mele_id1 ,
		inv_f.legal_entity_id1 ,
		mtminv.mtm_guid,
		SUM(inv_f.actual_quantity_32e) AS actual_quantity_32e ,
		SUM(inv_f.actual_quantity) AS actual_quantity ,
		MAX(latest.volume_uom_contract) AS volume_uom_contract ,
		MAX(inv_f.last_update_date) AS last_update_date ,
		MAX(inv_f.update_dt) AS update_dt ,
		MAX(latest.accounting_dt_id) AS accounting_dt_id ,
		MAX(latest.invoiced_status) AS invoiced_status ,
		MAX(latest.actual_shipment_dt_id) AS actual_shipment_dt_id ,
		MAX(latest.received_dt_id) AS received_dt_id,
		MAX(latest.json_object_id) AS json_object_id,
        MAX(c.price_status) AS price_status
	FROM
		ads.rs_inv_txns_f inv_f ,
		(
		SELECT
			*
		FROM
			ads.rs_inv_txns_f i
		WHERE
			source_system_id || cast(last_update_date AS VARCHAR(30)) || type = (
			SELECT
				max(source_system_id || cast(last_update_date AS VARCHAR(30)) || type)
			FROM
				ads.rs_inv_txns_f
			WHERE
				contract_type = i.contract_type
				AND contract_number = i.contract_number
				AND contract_line_number = i.contract_line_number
				AND release_number = i.release_number
				AND json_object_id not like '%BOD')
             AND i.json_object_id not like '%BOD'
			 AND i.contract_type IN ( 'SALE' , 'PURCHASE' )
			 AND i.type IN ( 'Actualized' , 'Reversed' )) latest ,
		ads.rs_mtm_calc_inv_txn_g mtminv ,
		ads.rs_contracts_d c
	WHERE
		inv_f.contract_type = latest.contract_type
		AND inv_f.contract_number = latest.contract_number
		AND inv_f.contract_line_number = latest.contract_line_number
		AND inv_f.release_number = latest.release_number
		and inv_f.source_system_id = latest.source_system_id
		AND inv_f.release_number = mtminv.release_number
		AND inv_f.json_object_id = mtminv.calc_key
		and inv_f.source_system_id = mtminv.mtl_txn_id
        AND inv_f.contract_line_id1 = c.surrogate_key2
		AND c.price_status = 'Final'
	GROUP BY inv_f.contract_type ,
		inv_f.contract_number ,
		inv_f.contract_line_number ,
		inv_f.release_number ,
		inv_f.source_system_id,
		mtminv.mtm_guid,
		inv_f.contract_line_id1 ,
		inv_f.location_id1 ,
		inv_f.product_id1 ,
		inv_f.mele_id1 ,
		inv_f.legal_entity_id1 ) inv ,
	ads.rs_product_d prod,
	ads.rs_mele_d MELE,
	ads.rs_int_org_d ORG
WHERE
	1 = 1
    AND mtm.mele_id=MELE.surrogate_key2(+)
	AND mtm.inventory_org_id=ORG.surrogate_key2(+)
	AND mtm.mtm_guid = inv.mtm_guid
	and inv.contract_number = mtm.contract_number
	and inv.contract_line_number = mtm.contract_line_number
	and inv.contract_type = mtm.contract_type
	AND inv.product_id1 = prod.surrogate_key2
	--AND mtm.current_flg = 'Y'
	AND mtm.mtm_type = 'Realized'
	AND mtm.bkr_txn_type = 'COMMODITY'
	AND mtm.currency_uom_contract = 'USD'
	--to be removed in later pahse

	 AND extract(year from TO_DATE(mtm.position_start_dt_id,'YYYYMMDD')) = '${v_year}'
  --  AND extract(month from TO_DATE(mtm.position_start_dt_id,'YYYYMMDD')) = '${v_month}'
group by 	inv.contract_type ,
	inv.contract_number ,
	inv.contract_line_number,
	inv.release_number,
	mtm.contract_line_id
; 