"""
High-Performance Bilingual E-Commerce Tabular Dataset Generator (20 Classes)
Generates realistic, meaningful, domain-grounded e-commerce tabular attributes.
Every single feature represents a real business metric (customer affinity, browsing history,
RFM scores, logistics, payment channels, and cross-category ratios) with ZERO random noise names.

Problem: E-Commerce Next-Purchase Department Prediction (20 Classes)
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# =====================================================================
# 20 E-COMMERCE TARGET DEPARTMENTS (BILINGUAL)
# =====================================================================
DEPARTMENTS = [
    {"id": 0,  "code": "mobiles",     "en": "Mobiles & Tablets",       "ar": "هواتف وأجهزة لوحية",    "price_mult": 1.8},
    {"id": 1,  "code": "electronics", "en": "Consumer Electronics",     "ar": "إلكترونيات استهلاكية",   "price_mult": 1.5},
    {"id": 2,  "code": "gaming",      "en": "Computers & Gaming",      "ar": "حواسيب وألعاب فيديو",   "price_mult": 2.2},
    {"id": 3,  "code": "kitchen",     "en": "Home & Kitchen",          "ar": "المنزل والمطبخ",        "price_mult": 0.9},
    {"id": 4,  "code": "appliances",  "en": "Large Appliances",        "ar": "أجهزة منزلية كبرى",     "price_mult": 3.0},
    {"id": 5,  "code": "men_fashion", "en": "Men's Fashion",           "ar": "أزياء رجالية",          "price_mult": 0.8},
    {"id": 6,  "code": "women_fash",  "en": "Women's Fashion",         "ar": "أزياء نسائية",          "price_mult": 1.1},
    {"id": 7,  "code": "baby",        "en": "Baby & Kids",             "ar": "أطفال ومواليد",         "price_mult": 0.7},
    {"id": 8,  "code": "beauty",      "en": "Beauty & Fragrances",     "ar": "عطور ومستحضرات تجميل",  "price_mult": 1.3},
    {"id": 9,  "code": "health",      "en": "Personal Care & Health",  "ar": "عناية شخصية وصحة",      "price_mult": 0.6},
    {"id": 10, "code": "groceries",   "en": "Supermarket & Groceries", "ar": "سوبرماركت وبقالة",      "price_mult": 0.4},
    {"id": 11, "code": "sports",      "en": "Sports & Fitness",        "ar": "رياضة ولياقة بدنية",    "price_mult": 1.0},
    {"id": 12, "code": "automotive",  "en": "Automotive & Hardware",   "ar": "سيارات ومعدات صيانة",   "price_mult": 1.2},
    {"id": 13, "code": "stationery",  "en": "Books & Stationery",      "ar": "كتب ومستلزمات مكتبية",  "price_mult": 0.3},
    {"id": 14, "code": "toys",        "en": "Toys & Games",            "ar": "ألعاب ترفيهية",          "price_mult": 0.5},
    {"id": 15, "code": "furniture",   "en": "Furniture & Decor",       "ar": "أثاث وديكور منزلي",     "price_mult": 2.5},
    {"id": 16, "code": "jewelry",     "en": "Watches & Jewelry",       "ar": "ساعات ومجوهرات",        "price_mult": 3.5},
    {"id": 17, "code": "pets",        "en": "Pet Supplies",            "ar": "مستلزمات حيوانات أليفة", "price_mult": 0.5},
    {"id": 18, "code": "office",      "en": "Office Electronics",      "ar": "تجهيزات مكتبية",         "price_mult": 1.6},
    {"id": 19, "code": "travel",      "en": "Travel & Luggage",        "ar": "حقائب ومستلزمات سفر",    "price_mult": 1.4}
]

DEPARTMENTS_EN = [d["en"] for d in DEPARTMENTS]
DEPARTMENTS_AR = [d["ar"] for d in DEPARTMENTS]
DEPT_CODES = [d["code"] for d in DEPARTMENTS]

# Categorical values
GOVERNORATES_AR = [
    "القاهرة", "الجيزة", "الإسكندرية", "الدقهلية", "الشرقية", "القليوبية",
    "الرياض", "مكة المكرمة", "المدينة المنورة", "القصيم", "المنطقة الشرقية",
    "عسير", "تبوك", "حائل", "دبي", "أبوظبي", "الشارقة", "عجمان"
]

PAYMENT_METHODS_AR = [
    "الدفع عند الاستلام",
    "بطاقة مدى",
    "بطاقة ائتمانية",
    "تابي (تقسيط)",
    "تمارا (تقسيط)",
    "محفظة رقمية",
    "سداد / فوري"
]

CUSTOMER_TIERS_AR = [
    "عميل جديد",
    "عميل فضي",
    "عميل ذهبي",
    "عميل بلاتيني",
    "عميل VIP"
]

SHIPPING_STATUS_AR = [
    "تم التوصيل بنجاح",
    "قيد التوصيل",
    "تم الشحن",
    "قيد التجهيز",
    "مرتجع",
    "ملغي من العميل"
]

DEVICE_TYPES_EN = ["iOS App", "Android App", "Mobile Safari", "Mobile Chrome", "Desktop Web", "Tablet Web"]
TRAFFIC_SOURCES_EN = ["Organic Search", "Google Ads", "Meta Ads", "TikTok Ads", "Direct", "Affiliate", "Email Newsletter", "Push Notification"]
COURIER_PARTNERS_EN = ["Aramex", "DHL Express", "SMSA Express", "Fetchr", "Bosta", "J&T Express", "In-House Fleet"]
FULFILLMENT_HUBS_EN = ["HUB-Cairo-East", "HUB-Cairo-West", "HUB-Alex", "HUB-Riyadh-North", "HUB-Jeddah-South", "HUB-Dubai-South"]


def generate_chunk(chunk_size, num_features=200, seed=42):
    """
    Generates a memory-efficient chunk where EVERY feature is a meaningful,
    interpretable e-commerce domain metric.
    """
    rng = np.random.default_rng(seed)
    
    # 1. Target Class: 20 E-commerce departments
    weights = rng.dirichlet(np.ones(20) * 1.5)
    target = rng.choice(20, size=chunk_size, p=weights)
    
    # Base dictionary of meaningful features
    feats = {}
    
    # Target and labels
    feats['target'] = target.astype(np.int8)
    feats['department_en'] = [DEPARTMENTS_EN[t] for t in target]
    feats['department_ar'] = [DEPARTMENTS_AR[t] for t in target]
    
    # 2. Core Bilingual Categorical Features
    feats['governorate_ar'] = rng.choice(GOVERNORATES_AR, size=chunk_size)
    feats['payment_method_ar'] = rng.choice(PAYMENT_METHODS_AR, size=chunk_size)
    feats['customer_tier_ar'] = rng.choice(CUSTOMER_TIERS_AR, size=chunk_size)
    feats['shipping_status_ar'] = rng.choice(SHIPPING_STATUS_AR, size=chunk_size)
    feats['device_type_en'] = rng.choice(DEVICE_TYPES_EN, size=chunk_size)
    feats['traffic_source_en'] = rng.choice(TRAFFIC_SOURCES_EN, size=chunk_size)
    feats['courier_en'] = rng.choice(COURIER_PARTNERS_EN, size=chunk_size)
    feats['fulfillment_hub_en'] = rng.choice(FULFILLMENT_HUBS_EN, size=chunk_size)
    
    # 3. Core Transaction & Session Metrics
    dept_price_mult = np.array([d['price_mult'] for d in DEPARTMENTS])
    order_val = rng.gamma(shape=2.5, scale=40.0, size=chunk_size) * dept_price_mult[target]
    discount_pct = rng.beta(a=2.0, b=5.0, size=chunk_size)
    cart_items = rng.poisson(lam=2.5, size=chunk_size) + 1
    session_sec = rng.exponential(scale=350.0, size=chunk_size) + 25.0
    clicks = (session_sec / 25.0 + rng.normal(0, 3, size=chunk_size)).clip(1, 150)
    customer_tenure_days = rng.integers(1, 1500, size=chunk_size)
    historical_orders = rng.negative_binomial(n=2, p=0.15, size=chunk_size)
    return_ratio = rng.beta(a=1.0, b=12.0, size=chunk_size)
    delivery_dist_km = rng.uniform(2.0, 1200.0, size=chunk_size)
    
    feats['order_value_sar'] = order_val.astype(np.float32)
    feats['discount_percentage'] = discount_pct.astype(np.float32)
    feats['cart_items_count'] = cart_items.astype(np.int16)
    feats['session_duration_seconds'] = session_sec.astype(np.float32)
    feats['click_count_in_session'] = clicks.astype(np.float32)
    feats['customer_tenure_days'] = customer_tenure_days.astype(np.int16)
    feats['historical_lifetime_orders'] = historical_orders.astype(np.int16)
    feats['historical_return_ratio'] = return_ratio.astype(np.float32)
    feats['delivery_distance_km'] = delivery_dist_km.astype(np.float32)
    
    # Core Ratios
    feats['avg_value_per_item_sar'] = (feats['order_value_sar'] / feats['cart_items_count']).astype(np.float32)
    feats['click_velocity_per_second'] = (feats['click_count_in_session'] / feats['session_duration_seconds']).astype(np.float32)
    feats['discount_savings_amount_sar'] = (feats['order_value_sar'] * feats['discount_percentage']).astype(np.float32)
    
    # 4. Temporal & Seasonal Features
    feats['session_hour'] = rng.integers(0, 24, size=chunk_size).astype(np.int8)
    feats['session_hour_sin'] = np.sin(2 * np.pi * feats['session_hour'] / 24.0).astype(np.float32)
    feats['session_hour_cos'] = np.cos(2 * np.pi * feats['session_hour'] / 24.0).astype(np.float32)
    feats['day_of_week'] = rng.integers(0, 7, size=chunk_size).astype(np.int8)
    feats['is_weekend_order'] = np.isin(feats['day_of_week'], [4, 5]).astype(np.int8)  # Fri-Sat in Middle East
    feats['is_payday_week_sar'] = rng.choice([0, 1], p=[0.75, 0.25], size=chunk_size).astype(np.int8)
    feats['is_ramadan_campaign'] = rng.choice([0, 1], p=[0.88, 0.12], size=chunk_size).astype(np.int8)
    feats['is_white_friday_sale'] = rng.choice([0, 1], p=[0.92, 0.08], size=chunk_size).astype(np.int8)
    
    # 5. Customer Financial & Payment Health
    feats['bnpl_installment_eligibility_score'] = rng.uniform(300, 850, size=chunk_size).astype(np.float32)
    feats['bnpl_past_installments_count'] = rng.poisson(1.5, size=chunk_size).astype(np.int16)
    feats['loyalty_reward_points_balance'] = rng.integers(0, 25000, size=chunk_size).astype(np.int32)
    feats['digital_wallet_balance_sar'] = rng.exponential(120.0, size=chunk_size).astype(np.float32)
    feats['free_shipping_gap_sar'] = np.maximum(0.0, 200.0 - feats['order_value_sar']).astype(np.float32)
    
    # 6. Browsing Depth & Intent
    feats['search_queries_in_session'] = rng.poisson(2.0, size=chunk_size).astype(np.int16)
    feats['zero_results_queries_count'] = (rng.binomial(2, 0.15, size=chunk_size)).astype(np.int8)
    feats['filter_price_slider_applied'] = rng.choice([0, 1], p=[0.6, 0.4], size=chunk_size).astype(np.int8)
    feats['filter_brand_selected_count'] = rng.poisson(1.0, size=chunk_size).astype(np.int8)
    feats['reviews_read_duration_seconds'] = rng.exponential(45.0, size=chunk_size).astype(np.float32)
    feats['product_image_swipes_count'] = rng.poisson(8.0, size=chunk_size).astype(np.int16)
    
    # 7. RFM & Customer Lifetime Value (CLV)
    feats['rfm_recency_days'] = rng.integers(1, 365, size=chunk_size).astype(np.int16)
    feats['rfm_frequency_orders'] = feats['historical_lifetime_orders']
    feats['rfm_monetary_lifetime_sar'] = (feats['order_value_sar'] * (feats['rfm_frequency_orders'] + 1)).astype(np.float32)
    feats['clv_predicted_365d_sar'] = (feats['rfm_monetary_lifetime_sar'] * 0.45 + rng.normal(100, 20, size=chunk_size)).astype(np.float32)
    feats['churn_risk_score'] = (1.0 / (1.0 + np.exp(-(feats['rfm_recency_days'] - 120.0) / 30.0))).astype(np.float32)
    
    # 8. Realistic, Leak-Free Department Signals (20 departments x 8 meaningful attributes = 160 features)
    # Customers shop across multiple categories with realistic overlaps and noise.
    # The target department has elevated signals, but NON-TARGET departments also have
    # organic views, past purchases, spend, wishlists, and search activity.
    for d_idx, d_meta in enumerate(DEPARTMENTS):
        code = d_meta['code']
        is_target_dept = (target == d_idx)
        
        # Views: Realistic baseline browsing across categories with probabilistic affinity boost
        view_base = rng.poisson(lam=1.8, size=chunk_size)
        target_has_view_surge = rng.choice([0, 1], p=[0.25, 0.75], size=chunk_size)
        view_boost = is_target_dept * target_has_view_surge * rng.poisson(lam=3.5, size=chunk_size)
        feats[f'dept_{code}_views_past_30d'] = (view_base + view_boost).astype(np.int16)
        
        # Cart additions past 30 days: naturally derived from views with realistic conversion rate
        cart_base = rng.binomial(feats[f'dept_{code}_views_past_30d'], 0.22)
        feats[f'dept_{code}_cart_adds_past_30d'] = cart_base.astype(np.int16)
        
        # Past orders in this category: Multi-category shoppers have orders across multiple categories
        orders_base = rng.poisson(lam=0.35, size=chunk_size)
        orders_target_boost = is_target_dept * rng.poisson(lam=0.65, size=chunk_size)
        feats[f'dept_{code}_orders_count_180d'] = (orders_base + orders_target_boost).astype(np.int16)
        
        # Total spend in category: proportional to orders plus realistic price spread
        spend_orders = feats[f'dept_{code}_orders_count_180d'] * d_meta['price_mult'] * rng.gamma(2.0, 25.0, size=chunk_size)
        spend_noise = (feats[f'dept_{code}_orders_count_180d'] > 0) * rng.uniform(5.0, 30.0, size=chunk_size)
        feats[f'dept_{code}_spend_total_sar_180d'] = (spend_orders + spend_noise).astype(np.float32)
        
        # Average dwell seconds: overlapping distributions
        dwell_base = rng.exponential(scale=18.0, size=chunk_size)
        dwell_boost = is_target_dept * rng.exponential(scale=25.0, size=chunk_size)
        feats[f'dept_{code}_avg_dwell_seconds'] = (dwell_base + dwell_boost).astype(np.float32)
        
        # Wishlist count: customers save items across various departments
        wishlist_base = rng.poisson(lam=0.25, size=chunk_size)
        wishlist_target_boost = is_target_dept * rng.poisson(lam=0.45, size=chunk_size)
        feats[f'dept_{code}_wishlist_items_count'] = (wishlist_base + wishlist_target_boost).astype(np.int8)
        
        # Discount sensitivity: customer behavioral trait (independent of specific target)
        feats[f'dept_{code}_discount_sensitivity'] = (
            discount_pct * rng.uniform(0.6, 1.2, size=chunk_size) + rng.uniform(0.05, 0.25, size=chunk_size)
        ).clip(0.0, 1.0).astype(np.float32)
        
        # Search queries intent score: continuous, overlapping Beta distributions on [0, 1]
        # Target has higher average intent (~0.48), non-target has lower average intent (~0.28)
        # but wide overlap so no simple threshold can separate them!
        intent_alpha = np.where(is_target_dept, 2.2, 1.4)
        intent_beta = np.where(is_target_dept, 2.5, 3.6)
        feats[f'dept_{code}_search_intent_score'] = rng.beta(intent_alpha, intent_beta).astype(np.float32)

    # 9. Cross-Department Spend and View Shares
    total_views = np.maximum(1, sum(feats[f'dept_{d["code"]}_views_past_30d'] for d in DEPARTMENTS))
    for d in DEPARTMENTS:
        code = d['code']
        feats[f'dept_{code}_view_share_ratio'] = (feats[f'dept_{code}_views_past_30d'] / total_views).astype(np.float32)
        
    # Check current feature count
    current_feat_count = len(feats) - 3  # minus target, department_en, department_ar
    needed = num_features - current_feat_count
    
    # 10. If more features requested (e.g. up to 2000 features), generate meaningful interaction ratios
    # between departments (e.g. cross-category ratio: mobiles vs accessories, groceries vs baby)
    if needed > 0:
        pair_count = 0
        for i in range(len(DEPARTMENTS)):
            if pair_count >= needed:
                break
            code_i = DEPARTMENTS[i]['code']
            v_i = feats[f'dept_{code_i}_views_past_30d']
            
            for j in range(i + 1, len(DEPARTMENTS)):
                if pair_count >= needed:
                    break
                code_j = DEPARTMENTS[j]['code']
                v_j = feats[f'dept_{code_j}_views_past_30d']
                
                # Meaningful pairwise cross-category interaction features
                feats[f'affinity_diff_{code_i}_vs_{code_j}'] = (v_i - v_j).astype(np.int16)
                pair_count += 1
                
                if pair_count < needed:
                    feats[f'affinity_ratio_{code_i}_to_{code_j}'] = ((v_i + 1.0) / (v_j + 1.0)).astype(np.float32)
                    pair_count += 1

                if pair_count < needed:
                    spend_i = feats[f'dept_{code_i}_spend_total_sar_180d']
                    spend_j = feats[f'dept_{code_j}_spend_total_sar_180d']
                    feats[f'spend_ratio_{code_i}_to_{code_j}'] = ((spend_i + 10.0) / (spend_j + 10.0)).astype(np.float32)
                    pair_count += 1

    df = pd.DataFrame(feats)
    return df


def generate_dataset(output_path, total_rows=3_000_000, num_features=200, chunk_size=100_000):
    """
    Streams generated chunks into a single compressed Apache Parquet file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print("=" * 80)
    print("REALISTIC BILINGUAL E-COMMERCE TABULAR DATASET GENERATOR")
    print("=" * 80)
    print(f"Target Output Path : {output_path}")
    print(f"Total Rows         : {total_rows:,}")
    print(f"Target Features    : {num_features:,}")
    print(f"Target Classes     : 20 Departments (Bilingual AR & EN)")
    print(f"Feature Domain     : 100% Meaningful E-Commerce Metrics (Views, Carts, Spend, Logistics, RFM)")
    print(f"Chunk Size         : {chunk_size:,} rows")
    print(f"Compression        : SNAPPY (Parquet)")
    print("-" * 80)
    
    start_total = time.time()
    num_chunks = int(np.ceil(total_rows / chunk_size))
    writer = None
    
    total_written = 0
    for i in range(num_chunks):
        current_chunk = min(chunk_size, total_rows - total_written)
        chunk_seed = 42 + i * 1337
        t0 = time.time()
        
        df_chunk = generate_chunk(current_chunk, num_features=num_features, seed=chunk_seed)
        
        table = pa.Table.from_pandas(df_chunk, preserve_index=False)
        
        if writer is None:
            writer = pq.ParquetWriter(
                output_path,
                table.schema,
                compression='snappy',
                use_dictionary=True
            )
            
        writer.write_table(table)
        total_written += current_chunk
        t_chunk = time.time() - t0
        
        pct = (total_written / total_rows) * 100
        print(f"Chunk {i+1:02d}/{num_chunks:02d} | Wrote {current_chunk:,} rows | "
              f"Progress: {pct:5.1f}% ({total_written:,}/{total_rows:,}) | Time: {t_chunk:.2f}s")
        
    if writer:
        writer.close()
        
    total_time = time.time() - start_total
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print("=" * 80)
    print("DATASET GENERATION COMPLETE")
    print(f"Output File : {output_path}")
    print(f"Total Rows  : {total_written:,}")
    print(f"File Size   : {file_size_mb:.2f} MB")
    print(f"Total Time  : {total_time:.2f} seconds ({total_written/total_time:.0f} rows/sec)")
    print("=" * 80)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_full = os.path.join(base_dir, "data", "ecommerce_20class_3m.parquet")
    default_sample = os.path.join(base_dir, "data", "ecommerce_20class_sample_50k.parquet")
    
    parser = argparse.ArgumentParser(description="Generate Bilingual 20-Class Tabular Dataset")
    parser.add_argument("--rows", type=int, default=3_000_000, help="Total number of rows (default: 3,000,000)")
    parser.add_argument("--features", type=int, default=200, help="Total number of features (default: 200)")
    parser.add_argument("--chunk-size", type=int, default=100_000, help="Chunk size for streaming (default: 100,000)")
    parser.add_argument("--output", type=str, default=default_full, help="Output Parquet path")
    parser.add_argument("--test-sample", action="store_true", help="Generate small test sample (50,000 rows)")
    
    args = parser.parse_args()
    
    if args.test_sample:
        generate_dataset(default_sample, total_rows=50_000, num_features=args.features, chunk_size=25_000)
    else:
        generate_dataset(args.output, total_rows=args.rows, num_features=args.features, chunk_size=args.chunk_size)
