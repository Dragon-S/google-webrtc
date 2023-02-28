#include "helper_mac.h"
#import <Foundation/Foundation.h>

namespace helper {

std::string GetXuanXingMeetDirMac() {
	NSArray *paths = NSSearchPathForDirectoriesInDomains(NSApplicationSupportDirectory,
                                                         NSUserDomainMask, YES);
	NSString *applicationSupportDirectory = [paths firstObject];
	std::string applicationSupportPath = std::string([applicationSupportDirectory UTF8String]);
	return applicationSupportPath + "/XuanXingMeet";
}

} // namespace helper
