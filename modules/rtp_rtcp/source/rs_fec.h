
#ifndef WEBRTC_MODULES_RTP_RTCP_SOURCE_RS_FEC_H_
#define WEBRTC_MODULES_RTP_RTCP_SOURCE_RS_FEC_H_

#define RS_FEC 
#define RS_FEC_PAYLOAD_TYPE 115

#define FEC_K  6
#define MAX_FEC_N  31
#define MAX_PAYLOAD_SIZE  1024
#define MAX_LIST_NUM  3
#define REG_SN_NUM  5

class RsFecEncoder
{
public:
	RsFecEncoder(void);
	~RsFecEncoder(void);
	int Init(uint8_t n,int sample_rate);
	void SetSampleRate(int sample_rate);
	int GetSampleRate() const;
	
	//int GetFrameSize() const;
	// void SetK(uint8_t k);
	void SetN(uint8_t n);
	void GetFecParameter(uint8_t &n, uint8_t &k);
	void SetFECStatus(bool enable);//If the parameter to 0, then close the FEC
	bool GetFECStatus() const;
	void SetPt(int8_t pt);
	int8_t GetPt() const;
	bool IsNeedSendRedPkg();
	

	int Encode(uint8_t *rtp_buf, int buf_len, const uint8_t *payload_data, int payload_size,int8_t pt,uint32_t ts );

	//int GetPreFrameData(uint8_t *pRTPBuf, int nBufLen, int nHeadLen);
private:
	//int AddRedDataToList(const uint8_t *pPayloadData, int len, int8_t pt, uint32_t ts);
	//int calcRedListLen(uint8_t nMulriple,uint8_t nOffset);
	int CreateFecHeader(uint8_t *buf, int pt, int index,int pkg_len);
	bool IsNeedAddRedData();
	uint8_t *GetEncodeMat(int n);
	void SetFrameSize(uint32_t ts,int payload_size);
	//int AddRedData(uint8_t *pBuf, int nBufLen);
	

	// fec_red_list *m_pFecRedListHeader;
	// uint8_t m_nDecimalMulripleQ4;
	// uint32_t m_fecFrameLoopCnt;
	int m_n_;
	int m_k_;
	int src_pkg_cnt_;
	int fec_pkg_idx_;
	int max_size_;
	int fec_size_; 
	uint8_t *enc_mat_[MAX_FEC_N-FEC_K];
	uint8_t *pkg_list_[MAX_FEC_N];
	bool enable_;
	//int8_t m_nRedCount;
	int8_t pt_;
	int sample_rate_;
	//int sample_rate_id_;
	int frame_size_;
	//int frame_size_id_;

	//bool need_set_n_;
	int  n_to_set_;
	bool fec_flag_;
	uint32_t ts_;
	//uint32_t ts_offset_;
	//CriticalSectionWrapper *_fecCritSect;
};


struct RsDecoderPara
{
	uint8_t n;
	uint8_t pt;
    uint16_t sn;
    uint32_t ts;
    uint16_t size;
    int pkg_num;
    bool is_decoded;
    int pkg_rd_cnt;
    int frame_size;
};


class RsFecDecoder
{
public:
	RsFecDecoder(void);
	~RsFecDecoder(void);
	int Init(void);
	// void SetSampleRate(int sample_rate);
	// void GetSampleRate() const;
	// void SetFrameSize(int size);
	// void GetFrameSize() const;
	// // void SetK(uint8_t k);
	// void SetN(uint8_t n);
	// void GetFecParameter(uint8_t &n, uint8_t &k);
	// void SetFECStatus(bool enable);//If the parameter to 0, then close the FEC
	// bool GetFECStatus() const;
    uint32_t GetLostRate();
	uint8_t GetPt(const uint8_t *payload_buf);

	int Decode(const uint8_t *payload_buf, int buf_size ,uint16_t sn , uint32_t ts, uint8_t *out_buf, int out_buf_len, uint8_t &out_pt);

	int GetRecoveryData(uint8_t *buf, int buf_len, uint16_t &sn , uint32_t &ts,uint8_t &pt);
private:
	//int AddRedDataToList(const uint8_t *pPayloadData, int len, int8_t pt, uint32_t ts);
	//int calcRedListLen(uint8_t nMulriple,uint8_t nOffset);
	int ParseFecHeader(const uint8_t *buf, uint8_t &pt,int &n,int &sid,int &rid,int &frame_size,int &src_size);
	// int ParseFecHeader(uint8_t *buf, int &n,int &sid,int &rid,int &frame_size,int &src_size);
	uint8_t *GetEncodeMat(int n);
	int AddPkgToList(const uint8_t *buf, int buf_size,uint8_t pt,int n,int id,uint16_t base_sn,uint32_t base_ts,int frame_size);
	void DeletePkgList(int index);
	
	//bool IsNeedAddRedData();
	//int AddRedData(uint8_t *pBuf, int nBufLen);
	


	int m_n_;
	// int m_k_;
	int list_cnt_;
	//int fec_pkg_idx_;

	

	uint8_t *enc_mat_[MAX_FEC_N-FEC_K];
	uint8_t *pkg_list_[MAX_LIST_NUM][FEC_K];
	int pkg_index_[MAX_LIST_NUM][FEC_K];
	RsDecoderPara *dec_para_[MAX_LIST_NUM];

	uint16_t decoded_sn_[REG_SN_NUM];
	uint32_t decoded_ts_[REG_SN_NUM];
	int decoded_sn_cnt_;
	//bool decoded_list_idx_[2];
	//int decoded_list_num_;

	//bool enable_;
	//int8_t m_nRedCount;
	//int8_t pt_;
	//int sample_rate_;
	//int sample_rate_id_;
	//int frame_size_;
	//int frame_size_id_;
	//CriticalSectionWrapper *_fecCritSect;
};

#endif 
